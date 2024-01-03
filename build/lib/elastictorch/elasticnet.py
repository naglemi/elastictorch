import torch
import torch.nn as nn

class ElasticNet(nn.Module):
    def __init__(self, n_features, alpha=1.0, l1_ratio=0.5):
        super(ElasticNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, x):
        return self.linear(x)

    def loss(self, pred, true):
        l1_loss = torch.norm(self.linear.weight, p=1)
        l2_loss = torch.norm(self.linear.weight, p=2)
        mse_loss = nn.functional.mse_loss(pred, true.view_as(pred))
        return mse_loss + self.alpha * (self.l1_ratio * l1_loss + (1-self.l1_ratio) * l2_loss)