import torch
import torch.nn as nn
from torch import optim
from src.losses import LossWrapper
from typing import List


__all__ = ["Controller"]


class Controller(object):
    def __init__(self, 
        loss_wrapper: LossWrapper, 
        model: nn.Module,
        mean_model: nn.Module,
        optimizer: optim.Optimizer
        ) -> None:
        self.loss_wrapper = loss_wrapper
        self.model = model
        self.mean_model = mean_model
        self.optimizer = optimizer


    def train_step(self, input: torch.Tensor, label: torch.Tensor, epoch: int, *args, **kwargs):
        """
        Define the training process for the model, easy for extension for multiple models

        Args:
            input (torch.Tensor): input tensor of the model
            label (torch.Tensor): ground truth of the input tensor

        Returns:
            loss (torch.FloatTensor): train loss
            loss_tuple (tuple[torch.FloatTensor]): a tuple of loss item
            output_no_grad (torch.FloatTensor): model output without grad
        """
        img1, img2, img3 = input[:, 0, ...].contiguous(), input[:, 1, ...].contiguous(), input[:, 2, ...].contiguous()
        self.optimizer.zero_grad()
        output1, output2, output3, en_output = self.model(img1, img2, img3)
        mean_output1, mean_output2, mean_output3 = self.mean_model(img1, img2, img3)

        loss, loss_tuple, outputs_no_grad = self.loss_wrapper([output1, output2, output3, en_output], [en_output.detach(), mean_output1, mean_output2, mean_output3, label], epoch=epoch, ema=False)
        loss.backward()
        self.optimizer.step()
        return loss, loss_tuple, outputs_no_grad


    def validate_step(self, input: torch.Tensor, label: torch.Tensor, epoch:int, *args, **kwargs):
        """
        Define the validation process for the model

        Args:
            input (torch.Tensor): input tensor for the model
            label (torch.Tensor): ground truth for the input tensor

        Returns:
            loss (torch.FloatTensor): validation loss item, without grad
            loss_tuple (tuple[torch.FloatTensor]): a tuple of loss item
            output_no_grad (torch.FloatTensor): model output without grad
        """
        img1, img2, img3 = input[:, 0, ...].contiguous(), input[:, 1, ...].contiguous(), input[:, 2, ...].contiguous()
        output1, output2, output3 = self.mean_model(img1, img2, img3)

        loss, loss_tuple, outputs_no_grad = self.loss_wrapper([output1, output2, output3], [label], ema=True, epoch=epoch)
        return loss.detach(), loss_tuple, outputs_no_grad