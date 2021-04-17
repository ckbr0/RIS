from typing import Callable, DefaultDict, Dict, Iterable, Union
import os
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from ignite.engine import Engine
from monai.engines import SupervisedTrainer
from monai.engines.utils import default_prepare_batch
from monai.handlers import LrScheduleHandler, TensorBoardStatsHandler, CheckpointSaver, StatsHandler

from handlers import ValidationHandler, MetricLogger
from validator import Validator

class Trainer(SupervisedTrainer):

    def __init__(
        self,
        device: torch.device,
        out_dir,
        out_name,
        max_epochs: int,
        train_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        validator: Engine,
        lr_scheduler = None,
        summary_writer: SummaryWriter = None,
        validation_epoch: int = 1,
        validation_interval: int = 1,
        post_transform = None,
        non_blocking: bool = False,
        amp: bool = False,
    ) -> None:
        
        self.validation_epoch = validation_epoch
        self.validation_interval = validation_interval
        self.validator = validator
        self.out_dir = out_dir
        self.out_name = out_name
        self.summary_writer = summary_writer
        self.lr_scheduler = lr_scheduler

        super().__init__(
            device,
            max_epochs,
            train_data_loader,
            network,
            optimizer,
            loss_function,
            epoch_length=None,
            non_blocking=non_blocking,
            iteration_update=self._iteration,
            inferer=None,
            post_transform=post_transform,
            amp=amp
        )

    def run(self, date=None) -> str:
        
        if date is not None:
            now = date
        else:
            now = datetime.datetime.now()
        datetime_string = now.strftime('%d/%m/%Y %H:%M:%S')
        print(f'Training started: {datetime_string}')

        now = datetime.datetime.now()
        timedate_info = str(now).split(' ')[0] + '_' + str(now.strftime("%H:%M:%S")).replace(':', '-')
        training_dir = os.path.join(self.out_dir, 'training')
        if not os.path.exists(training_dir):
            os.mkdir(training_dir)
        self.output_dir = os.path.join(training_dir, self.out_name +  '_' + timedate_info)
        os.mkdir(self.output_dir)
        
        self.validator.output_dir = self.output_dir

        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(log_dir=self.output_dir)
        if self.validator.summary_writer is None:
            self.validator.summary_writer = self.summary_writer

        handlers = [
            MetricLogger(self.output_dir, validator=self.validator),
            ValidationHandler(
                validator=self.validator,
                start=self.validation_epoch,
                interval=self.validation_interval
            ),
            StatsHandler(tag_name="loss", output_transform=lambda x: x["loss"]),
            TensorBoardStatsHandler(
                summary_writer=self.summary_writer,
                tag_name="Loss",
                output_transform=lambda x: x["loss"]
            ),
        ]
        save_dict = { 'network': self.network, 'optimizer': self.optimizer }
        if self.lr_scheduler is not None:
            handlers.insert(0, LrScheduleHandler(lr_scheduler=self.lr_scheduler, print_lr=True))
            save_dict['lr_scheduler'] = self.lr_scheduler
        handlers.append(
            CheckpointSaver(save_dir=self.output_dir, save_dict=save_dict, save_interval=1, n_saved=1)
        )
        self._register_handlers(handlers)

        super().run()
        return self.output_dir

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        return super()._iteration(engine, batchdata)
