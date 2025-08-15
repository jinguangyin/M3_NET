# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:49:19 2025

@author: HP
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.ff = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, x):
        return self.ff(x)+x
    
   
class FFNMoE(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.w_gate = nn.Linear(hidden_dim, num_experts)

        self.experts = nn.ModuleList(
            [
                FeedForward(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: Tensor):

        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]
        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)  # (batch_size, seq_len, output_dim, num_experts)
        # Combine expert outputs and gating scores
        moe_output = torch.sum(gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1)

        return moe_output, gate_scores


class S_MLPMixerBlock(nn.Module):
    
    def __init__(self, num_patches, hidden_dim, channel_dim):
        super(S_MLPMixerBlock, self).__init__()
        
        self.token_mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches*4),
            nn.GELU(),
            nn.Linear(num_patches*4, num_patches)
        )
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, channel_dim),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(channel_dim, hidden_dim)
        )
        self.channel_mixer = FFNMoE(hidden_dim=hidden_dim, num_experts=4)
        # self.ln1 = nn.LayerNorm(hidden_dim)
        # self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, adj):
        origin_input = x
        x = torch.einsum('ng,bnd->bgd', adj, x)
        y = x.transpose(1, 2) 
        y = self.token_mixer(y)
        y = y.transpose(1, 2)
        y = torch.einsum('gn,bgd->bnd', adj.mT, y)
        x = origin_input + y
        
        # Channel mixing
        y,_ = self.channel_mixer(x)
        x = x + y
        return x
  
    
class M3Net(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.num_group = model_args["num_group"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.encoder = nn.ModuleList()

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        
        self.group = nn.Parameter(torch.randn(self.num_nodes, self.num_group))

        self.time_series_emb_layer = nn.Linear(self.input_dim*self.input_len , self.embed_dim)
     
        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)

        #self.hidden_dim = self.embed_dim+self.node_dim * int(self.if_spatial)
        
        for i in range(self.num_layer):
            self.encoder.append(S_MLPMixerBlock(self.num_group, self.hidden_dim, self.hidden_dim))         


        self.regression_layer = nn.Linear(self.hidden_dim, 12)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        batch_size, input_len, num_nodes, _ = input_data.shape
        # print('his_data_shape', history_data.shape)
        # print('input_data_shape', input_data.shape)

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
            #time_in_day_emb = time_in_day_emb.unsqueeze(1).expand(-1, input_len, -1,-1)
            # print('time_in_day_emb_shape', time_in_day_emb.shape)
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
            #day_in_week_emb = day_in_week_emb.unsqueeze(1).expand(-1, input_len, -1,-1)
            # print('day_in_week_emb_shape', day_in_week_emb.shape)
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1)
        time_series_emb = self.time_series_emb_layer(input_data)
        # print('time_series_emb_shape',time_series_emb.shape)

        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1,-1))        
            # print('node_emb_shape', node_emb[0].shape)
        # temporal embeddings
        tem_emb = []
        tem_emb.append(time_in_day_emb)
        tem_emb.append(day_in_week_emb)

        # concate all embeddings

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)

        group_adj = F.softmax(self.group, dim=1)

        # skip = 0
        for i in range(self.num_layer):
            hidden = self.encoder[i](hidden, group_adj)

        prediction = self.regression_layer(hidden)        
        prediction = prediction.transpose(1, 2).unsqueeze(-1)

        return prediction