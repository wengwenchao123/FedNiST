"""Node-wise dense MoE with exactly one residual connection."""
import torch
from torch import nn


class ResidualMixtureOfExperts(nn.Module):
    def __init__(self, channels, num_experts=4, dropout=0.15):
        super().__init__()
        if num_experts < 1:
            raise ValueError('num_experts must be positive')
        self.gate = nn.Conv2d(channels, num_experts, kernel_size=1)
        # Keep FedNiST's ReLU/dropout and width; experts have no inner residual.
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(),
                          nn.Dropout(dropout), nn.Conv2d(channels, channels, 1))
            for _ in range(num_experts)
        ])

    def routing_weights(self, x):
        return torch.softmax(self.gate(x), dim=1)

    def forward(self, x):
        weights = self.routing_weights(x)  # [B, K, N, 1], no node mixing
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return x + (weights.unsqueeze(2) * outputs).sum(dim=1)
