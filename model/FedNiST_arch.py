import torch
from torch import nn
import torch.nn.functional as F
from .mlp import MultiLayerPerceptron
from .moe import ResidualMixtureOfExperts




class FedNiST(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args["num_nodes"]

        self.if_spatial = model_args["if_node"]
        self.node_dim = model_args["node_dim"]
        self.global_model = global_model(**model_args)
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]


        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size*self.day_of_week_size, self.temp_dim_tid))
            # nn.init.xavier_uniform_(self.time_in_day_emb)
            nn.init.zeros_(self.time_in_day_emb)

    def used_delta_l2_loss(self, local_time_emb: torch.Tensor) -> torch.Tensor:

        used_delta = local_time_emb * self.local_scale
        return (used_delta ** 2).mean()

    def forward(self, history_data: torch.Tensor, client_node_id=None,epoch=None) -> torch.Tensor:
        batch_size= history_data.shape[0]

        node_emb = []
        if self.if_spatial:
            # if client_node_id != None:
            #     # expand node embeddings
            #     node_emb.append(self.node_emb[client_node_id].unsqueeze(0).expand(
            #         batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
            # else:
            selected_nodes = self.node_emb[self.client_node_ids] if hasattr(self, 'client_node_ids') else self.node_emb
            node_emb.append(selected_nodes.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))


        if self.if_time_in_day:
            t_i_d_data = history_data[..., -2]
            d_i_w_data = history_data[..., -1]


            local_time_in_day_emb = self.time_in_day_emb[
                ((t_i_d_data[:, -1, :]+d_i_w_data[:, -1, :]) * self.time_of_day_size).type(torch.LongTensor)]

        else:
            local_time_in_day_emb = None


        prediction = self.global_model(history_data,node_emb,local_time_in_day_emb,epoch)

        client_emb = selected_nodes.mean(dim=0)
        client_emb = F.normalize(client_emb, dim=0)
        if self.training:
             return prediction, client_emb
        else:
            return prediction
        # return prediction
class global_model(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        # self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.out_dim = model_args["out_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        # self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]


        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size*self.day_of_week_size,self.temp_dim_tid))# self.temp_dim_tid*2))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) \
           # + self.temp_dim_diw*int(self.if_day_in_week)

        # self.encoder = nn.Sequential(
        #     *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        self.spatial_enconder = MultiLayerPerceptron(self.embed_dim+self.node_dim,
                                               self.embed_dim+self.node_dim)
        self.s_encoder = MultiLayerPerceptron(self.embed_dim+self.node_dim +self.temp_dim_tid,
                                               self.embed_dim+self.node_dim +self.temp_dim_tid)
        self.se_encoder = MultiLayerPerceptron(self.embed_dim+self.node_dim +self.temp_dim_tid*2,
                                               self.embed_dim+self.node_dim +self.temp_dim_tid*2)
        self.es_encoder = MultiLayerPerceptron(self.embed_dim+self.node_dim +self.temp_dim_tid*3,
                                               self.embed_dim+self.node_dim +self.temp_dim_tid*3)
        self.last_block = model_args.get('last_block', 'mlp')
        self.num_experts = model_args.get('num_experts', 4)
        if self.num_experts < 1:
            raise ValueError('num_experts must be positive')
        final_width = self.embed_dim + self.node_dim + self.temp_dim_tid * 4
        if self.last_block in ('moe', 'all_moe'):
            self.ef_encoder = ResidualMixtureOfExperts(final_width, num_experts=self.num_experts)
        elif self.last_block == 'mlp':
            self.ef_encoder = MultiLayerPerceptron(final_width, final_width)
        else:
            raise ValueError('last_block must be mlp, moe or all_moe')
        if self.last_block == 'all_moe':
            for index, name in enumerate(('s_encoder', 'se_encoder', 'es_encoder'), start=1):
                width = self.embed_dim + self.node_dim + self.temp_dim_tid * index
                setattr(self, name, ResidualMixtureOfExperts(width, num_experts=self.num_experts))
        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.embed_dim+self.node_dim +self.temp_dim_tid*4, out_channels=self.output_len*self.out_dim, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, node_emb=None,local_time_in_day_emb=None,epoch=None) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        B, L, N, C =history_data.shape
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., -2]
            d_i_w_data = history_data[..., -1]
            # In the datasets used in STID, the time_of
            # _day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            # time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
            global_history_start_time_in_day_emb = self.time_in_day_emb[
                ((t_i_d_data[:, 0, :]+d_i_w_data[:, 0, :]) * self.time_of_day_size).type(torch.LongTensor)].transpose(1, 2).unsqueeze(-1)
            global_history_end_time_in_day_emb = self.time_in_day_emb[
                ((t_i_d_data[:, -1, :]+d_i_w_data[:, -1, :]) * self.time_of_day_size).type(torch.LongTensor)].transpose(1, 2).unsqueeze(-1)
            # Convert to an integer slot before advancing; wrap over the full week.
            history_end_index = ((t_i_d_data[:, -1, :] + d_i_w_data[:, -1, :]) * self.time_of_day_size).long()
            week_steps = self.time_of_day_size * self.day_of_week_size
            global_future_start_time_in_day_emb = self.time_in_day_emb[
                (history_end_index + 1) % week_steps].transpose(1, 2).unsqueeze(-1)

            global_future_end_time_in_day_emb = self.time_in_day_emb[
                (history_end_index + self.output_len) % week_steps].transpose(1, 2).unsqueeze(-1)

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)



        hidden = torch.cat([time_series_emb] + node_emb, dim=1)
        hidden = self.spatial_enconder(hidden)

        hidden = torch.cat([hidden,global_history_start_time_in_day_emb], dim=1)
        hidden = self.s_encoder(hidden)

        hidden = torch.cat([hidden,global_history_end_time_in_day_emb], dim=1)
        hidden = self.se_encoder(hidden)
        #
        hidden = torch.cat([hidden,global_future_start_time_in_day_emb], dim=1)
        hidden = self.es_encoder(hidden)
        #
        hidden = torch.cat([hidden,global_future_end_time_in_day_emb], dim=1)
        hidden = self.ef_encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden).reshape(B,self.output_len,-1,self.out_dim)


        return prediction
