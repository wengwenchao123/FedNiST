import copy
import torch
class Server:

    def __init__(self, model, device='cpu', aggregation='fedavg', projection_seed=0):
        self.global_model = model.to(device)
        self.last_aggregation_stats = None
        self.aggregation = 'fedavg'

    @torch.no_grad()
    def aggregate(self, client_weights_list, client_emb_list=None):
        """
        等权聚合；FedGH 先对整轮共享参数增量做冲突投影。
        client_emb_list 仅为兼容现有调用保留，不参与聚合。
        """
        if not client_weights_list:
            raise ValueError('Cannot aggregate an empty client list')
        num_clients = len(client_weights_list)
        avg_weights = copy.deepcopy(client_weights_list[0])
        for key in avg_weights.keys():
            avg_weights[key] = client_weights_list[0][key] / num_clients
            for i in range(1, num_clients):
                avg_weights[key] += client_weights_list[i][key] / num_clients
        self.global_model.load_state_dict(avg_weights)

    def get_weights(self):
        return {key: value.detach().clone() for (key, value) in self.global_model.state_dict().items()}