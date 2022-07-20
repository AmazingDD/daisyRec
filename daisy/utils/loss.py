import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -(self.gamma + torch.sigmoid(pos_score - neg_score)).log().sum()

        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        loss = torch.clamp(1 - (pos_score - neg_score), min=0).sum()

        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, pos_score, neg_score):
        loss = (neg_score - pos_score).sigmoid().sum() + neg_score.pow(2).sigmoid().sum()

        return loss
