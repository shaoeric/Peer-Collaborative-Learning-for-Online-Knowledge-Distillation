import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from easydict import EasyDict
from .utils import NetIO
from .losses import LossWrapper
from .optimizer import OptimizerBuilder
from .schemes import SchedulerBuilder
from .utils import LoggerBuilder, AverageMeter, update_ema_variables
from .metrics import MetricBuilder
from .controller import Controller


__all__ = ['Trainer']


class Trainer:
    def __init__(self, 
        config: EasyDict,
        model: nn.Module,
        mean_model: nn.Module,
        wrapper: LossWrapper,
        ioer: NetIO,
        *args, **kwargs
    ) -> None:
        
        self.config = config
        self.loss_wrapper = wrapper
        self.ioer = ioer
        self.logger, self.summary = None, None

        self.__parse_config()
        
        self.metric_func_list = self.__get_metrics()

        self.__num_loss = self.loss_wrapper.num_meter
        self.__num_metric = len(self.metric_func_list)

        self.model = model
        self.mean_model = mean_model
        (self.optimizer, self.optimizer_params), (self.scheduler, self.scheduler_params) = self.__build_optimizer(self.model, self.lr)

        self.controller = Controller(loss_wrapper=self.loss_wrapper, model=self.model, mean_model=self.mean_model, optimizer=self.optimizer)
  
        self.logger, self.summary = LoggerBuilder(config).load()
        if self.logger is not None:
            self.logger.info(self.optimizer_params)
            self.logger.info(self.scheduler_params)
        
        self.__global_step = 0

    def __parse_config(self):
        self.max_epoch = self.config.train["epochs"]
        self.lr = self.config.train["lr"]
        self.loss_names = self.config.train['criterion']['names']
        self.metric_names = self.config.train["metric"]["names"]
        self.key_metric_name = self.config.train["metric"]["key_metric_name"]

        self.log_step_freq = self.config.output["log_step_freq"]
        self.log_epoch_freq = self.config.output["log_epoch_freq"]

    def __build_optimizer(self, model: nn.Module, lr: float, *args, **kwargs):
        optimizer_name = self.config.train.optimizer
        scheduler_name = self.config.train.schedule
        
        optimizer, optimizer_config = OptimizerBuilder.load(optimizer_name, model.parameters(), lr)
        scheduler, scheduler_config = SchedulerBuilder.load(scheduler_name, optimizer, self.max_epoch)
        return (optimizer, optimizer_config), (scheduler, scheduler_config)


    def __get_metrics(self):
        metric_func_list = []

        for metric_name in self.metric_names:
            metric_func = MetricBuilder.load(metric_name)
            metric_func_list.append(metric_func)
        return metric_func_list

    
    def train(self, epoch: int, dataloader: DataLoader):
        if self.logger is not None:
            self.logger.info("Training epoch [{} / {}]".format(epoch, self.max_epoch))

        use_cuda = torch.cuda.is_available()
        self.model.train()
        self.mean_model.train()

        current_lr = self.scheduler.get_lr()[0]

        if self.summary is not None:
            self.summary.add_scalar("train/lr", current_lr, epoch)

        loss_recorder = AverageMeter(type='scalar', name='total loss')
        loss_list_recorder = AverageMeter(type='tuple', num_scalar=3, names=['model1', 'model2', 'model3'])
        metric_list_recorder1 = AverageMeter(type='tuple', num_scalar=self.__num_metric, names=self.metric_names)
        metric_list_recorder2 = AverageMeter(type='tuple', num_scalar=self.__num_metric, names=self.metric_names)
        metric_list_recorder3 = AverageMeter(type='tuple', num_scalar=self.__num_metric, names=self.metric_names)
    
        # === current epoch begins training ===
        for batch_idx, data in enumerate(dataloader):
            img, target = data
            img = img.float()
            target = target.long()
            batch_size = img.size(0)
            if use_cuda:
                img = img.cuda()
                target = target.cuda()

            loss, loss_tuple, outputs_no_grad = self.controller.train_step(img, target, epoch)
            update_ema_variables(self.model, self.mean_model, alpha=0.999,  global_step=self.__global_step)

            loss_recorder.update(loss.item(), batch_size)

            loss_list_recorder.update(loss_tuple, batch_size)

            metrics = tuple([func(outputs_no_grad[0], target) for func in self.metric_func_list])
            metric_list_recorder1.update(metrics, batch_size)

            metrics = tuple([func(outputs_no_grad[1], target) for func in self.metric_func_list])
            metric_list_recorder2.update(metrics, batch_size)

            metrics = tuple([func(outputs_no_grad[2], target) for func in self.metric_func_list])
            metric_list_recorder3.update(metrics, batch_size)

            if self.log_step_freq > 0 and self.__global_step % self.log_step_freq == 0:
                if self.logger:
                    msg = "[Train] Epoch:[{}/{}] batch:[{}/{}] loss: {:.4f} loss list: {} metric list: {}".format(epoch, self.max_epoch, batch_idx + 1, len(dataloader),
                    loss_recorder.get_value(), loss_list_recorder, (metric_list_recorder1, metric_list_recorder2, metric_list_recorder3))
                    self.logger.info(msg)

            self.__global_step += 1
            
        # === current epoch finishes training ===

        if epoch % self.log_epoch_freq == 0:
            if self.logger:
                msg = "[Train] Epoch:[{}/{}] loss: {:.4f} loss list: {} metric list: {}".format(epoch, self.max_epoch, loss_recorder.get_value(), loss_list_recorder, (metric_list_recorder1, metric_list_recorder2, metric_list_recorder3))
                self.logger.info(msg)
            if self.summary:
                self.summary.add_scalar("train/epoch_loss", loss_recorder.get_value(), epoch)
                names = ["model1 acc", "model2 acc", "model3 acc"]
                values = [metric_list_recorder1.meter.get_value()[0], metric_list_recorder2.meter.get_value()[0], metric_list_recorder3.meter.get_value()[0]]
                for name, value in zip(names, values):
                    self.summary.add_scalar("train/epoch_{}".format(name), value, epoch)

        self.scheduler.step()
        
    def validate(self, epoch: int, dataloader: DataLoader):
        self.mean_model.eval()

        loss_recorder = AverageMeter(type="scalar", name='total loss')
        loss_list_recorder = AverageMeter(type="tuple", num_scalar=3, names=['model1', 'model2', 'model3'])
        metric_list_recorder1 = AverageMeter(type='tuple', num_scalar=self.__num_metric, names=self.metric_names)
        metric_list_recorder2 = AverageMeter(type='tuple', num_scalar=self.__num_metric, names=self.metric_names)
        metric_list_recorder3 = AverageMeter(type='tuple', num_scalar=self.__num_metric, names=self.metric_names)
        use_cuda = torch.cuda.is_available()
        val_step = 0
        with torch.no_grad():
            # === current epoch begins validation ===
            for batch_idx, data in enumerate(dataloader):
                img, target = data
                img = img.float()
                target = target.long()
                batch_size = img.size(0)
                if use_cuda:
                    img = img.cuda()
                    target = target.cuda()
                loss, loss_tuple, output_no_grad = self.controller.validate_step(img, target, epoch)
                
                loss_recorder.update(loss.item(), batch_size)
                loss_list_recorder.update(loss_tuple, batch_size)
                metrics = tuple([func(output_no_grad[0], target) for func in self.metric_func_list])
                metric_list_recorder1.update(metrics, batch_size)

                metrics = tuple([func(output_no_grad[1], target) for func in self.metric_func_list])
                metric_list_recorder2.update(metrics, batch_size)
                
                metrics = tuple([func(output_no_grad[2], target) for func in self.metric_func_list])
                metric_list_recorder3.update(metrics, batch_size)
            
                if self.log_step_freq > 0 and val_step % self.log_step_freq == 0:
                    if self.logger:
                        msg = "[Validation] Epoch:[{}/{}] batch:[{}/{}] loss: {:.4f} loss list: {} metric list: {}".format(epoch, self.max_epoch, batch_idx + 1, len(dataloader),
                        loss_recorder.get_value(), loss_list_recorder, (metric_list_recorder1, metric_list_recorder2, metric_list_recorder3))
                        self.logger.info(msg)
                val_step += 1
            # === current epoch finishes validation === 

            if epoch % self.log_epoch_freq == 0:
                if self.logger:
                    msg = "[Validation] Epoch:[{}/{}] loss: {:.4f} loss list: {} metric list: {}".format(epoch, self.max_epoch, loss_recorder.get_value(), loss_list_recorder, (metric_list_recorder1, metric_list_recorder2, metric_list_recorder3))
                    self.logger.info(msg)
                if self.summary:
                    self.summary.add_scalar("val/epoch_loss", loss_recorder.get_value(), epoch)
                    names = ["model1 acc", "model2 acc", "model3 acc"]
                    values = [metric_list_recorder1.meter.get_value()[0], metric_list_recorder2.meter.get_value()[0], metric_list_recorder3.meter.get_value()[0]]
                    for name, value in zip(names, values):
                        self.summary.add_scalar("val/epoch_{}".format(name), value, epoch)

            check_metric = max(values)
            # save checkpoint referring to the save_freq and the saving strategy, besides record the key metric value
            self.ioer.save_file(self.mean_model, epoch, check_metric)