import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import copy


class Server:
    def __init__(self, model,device='cpu'):
        self.global_model = model.to(device)
        self.device = device

    def aggregate(self, client_weights_list,client_emb_list):
        """
        FedAvg 核心：对参数取平均
        """
        # 拿出第一个客户端的参数作为基准
        client_embs  =torch.stack(client_emb_list,dim=0)

        global_client_emb = client_embs.mean(dim=0, keepdim=True)  # [1, 32]
        global_client_emb = F.normalize(global_client_emb, p=2, dim=1)

        scores = F.cosine_similarity(client_embs, global_client_emb, dim=1)  # [K]

        attn_weights = F.softmax(scores, dim=0)

        avg_weights = copy.deepcopy(client_weights_list[0])

        # 逐层累加其他客户端的参数
        for key in avg_weights.keys():
            avg_weights[key] = attn_weights[0] * client_weights_list[0][key]
            for i in range(1, len(client_weights_list)):
                avg_weights[key] += attn_weights[i] * client_weights_list[i][key]

        # # 更新全局模型
        self.global_model.load_state_dict(avg_weights)

    def get_weights(self):
        return self.global_model.state_dict()


# --- 2. 定义客户端 (Client) ---
class Client:
    def __init__(self, model, client_id, data_loader, client_node_id, lr, weight_decay, device='cpu'):
        self.client_id = client_id
        self.train_data_loader = data_loader[0]
        self.val_data_loader = data_loader[1]
        self.test_data_loader = data_loader[2]
        self.client_node_id = client_node_id
        self.device = device
        self.model = model.to(self.device)
        self.client_loss = np.Inf
        self.client_test_loss = np.Inf

    def local_train(self, global_weights, epochs=1):
        # 加载服务器发来的参数
        self.model.load_state_dict(global_weights)


        self.model.train()

        for epoch in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        # 关键：只返回参数 (state_dict)，不返回数据！
        return copy.deepcopy(self.model.state_dict())
