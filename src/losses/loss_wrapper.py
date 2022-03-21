import torch
import torch.nn as nn
from typing import List
import math

class LossWrapper(nn.Module):
    def __init__(self, loss_func_list: List[nn.Module], loss_weight_list: List[float], *args, **kwargs) -> None:
        super(LossWrapper, self).__init__()
        self.loss_func_list = loss_func_list
        self.loss_weight_list = loss_weight_list
        assert len(self.loss_func_list) == len(self.loss_weight_list), "length of loss function list should match the length of loss weight list"
        self.num_meter = len(self.loss_func_list)
        if len(self.loss_func_list) == 1:
            self.loss_weight_list = [1.0]
        
        self.ce, self.kd, self.pm = self.loss_func_list
    
    def compute_rampup_weight(self, epoch, lambd=1.0, alpha=80):
        if epoch > alpha:
            return lambd
        else:
            return lambd * math.exp(-5 * (1 - epoch / lambd) ** 2)

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], epoch, *args, **kwargs):
        """
        Calculate the total loss between model prediction and target list
        
        Args:
            pred (torch.Tensor): a list of model prediction
            targets (List[torch.Tensor]): a list of targets for multi-task / multi loss training

        Returns:
            loss (torch.FloatTensor): a weighted loss tensor
            loss_list (tuple[torch.FloatTensor]): a tuple of loss item
            pred (torch.FloatTensor): model output without grad
        """
        if not kwargs['ema']:
            en_label, mean_label1, mean_label2, mean_label3, label = targets
            pred1, pred2, pred3, pred_en = preds
            
            loss = self.ce(pred_en, label)
            rampup_weight = self.compute_rampup_weight(epoch)

            # model1
            loss_ce = self.ce(pred1, label)
            loss_pe = self.kd(pred1, en_label) * rampup_weight
            loss_pm = self.pm(pred1, [mean_label2, mean_label3]) * rampup_weight
            loss1 = loss_ce + loss_pe + loss_pm 
            # model2
            loss_ce = self.ce(pred2, label)
            loss_pe = self.kd(pred2, en_label) * rampup_weight
            loss_pm = self.pm(pred2, [mean_label1, mean_label3]) * rampup_weight
            loss2 = loss_ce + loss_pe + loss_pm 
            # model3
            loss_ce = self.ce(pred3, label)
            loss_pe = self.kd(pred3, en_label) * rampup_weight
            loss_pm = self.pm(pred3, [mean_label1, mean_label2]) * rampup_weight
            loss3 = loss_ce + loss_pe + loss_pm

            loss += loss1 + loss2 + loss3
            return loss, (loss1.detach(), loss2.detach(), loss3.detach()), (mean_label1, mean_label2, mean_label3)

        else:
            label = targets[0]
            pred1, pred2, pred3 = preds
            loss1 = self.ce(pred1, label)
            loss2 = self.ce(pred2, label)
            loss3 = self.ce(pred3, label)
            loss = loss1 + loss2 + loss3
            return loss.detach(), (loss1.detach(), loss2.detach(), loss3.detach()), (pred1, pred2, pred3)

if __name__ == '__main__':
    from loss_builder import LossBuilder
    model = nn.Linear(3, 5)
    x = torch.randn(2, 3)
    y = torch.randint(0, 5, size=(2, ))

    # loss = CrossEntropyLoss * 1.0
    ce_loss = LossBuilder.load("CrossEntropyLoss")
    wrapper = LossWrapper([ce_loss], [1.0])
    out = model(x)
    loss, loss_list, output = wrapper.forward(out, [y])
    print("loss: {} loss_list: {}, pred: {}".format(loss, loss_list, output.max(dim=1)[1]))