import imp
from turtle import forward


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['KDLoss']

class KDLoss(nn.Module):
    def __init__(self, temperature=3.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.func = nn.KLDivLoss(reduction='batchmean')


    def forward(self, pred, target):
        """_summary_

        Parameters
        ----------
        pred : _type_  [N, C]
            _description_
        target : _type_  [N, C]
            _description_
        """
        loss = self.func(F.log_softmax(pred / self.temperature, dim=1), F.softmax(target / self.temperature, dim=1)) * self.temperature * self.temperature
        return loss
 

class PMLoss(nn.Module):
    def __init__(self, temperature=3.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.kd = KDLoss(temperature=self.temperature)

    def forward(self, pred, targets):
        loss = 0
        for i in range(len(targets)):
            target = targets[i].detach()
            pmloss = self.kd(pred, target)
            loss += pmloss
        return loss / len(targets)