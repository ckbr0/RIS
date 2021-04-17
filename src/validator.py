from typing import Dict, Iterable, Union, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.metrics import Accuracy, EpochMetric, Loss
from monai.engines import SupervisedEvaluator
from monai.transforms import Transform, AsDiscrete
from monai.utils import ForwardMode
from monai.handlers import TensorBoardStatsHandler, CheckpointSaver, StatsHandler, MetricsSaver, ROCAUC, EarlyStopHandler

class Validator(SupervisedEvaluator):

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        loss_function,
        n_classes,
        patience = 20,
        summary_writer: SummaryWriter = None,
        non_blocking: bool = False,
        post_transform: Optional[Transform] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
        ) -> None:
        self.summary_writer = summary_writer
        self.early_stop_handler = EarlyStopHandler(patience=patience, score_function=lambda engine: engine.state.metrics[engine.state.key_metric_name])

        if n_classes > 1:
            to_onehot = AsDiscrete(to_onehot=True, n_classes=2)
        else:
            to_onehot = lambda x: x
        
        super().__init__(
            device,
            val_data_loader,
            network,
            non_blocking=non_blocking,
            iteration_update=self._iteration,
            post_transform=post_transform,
            key_val_metric={
                "Valid_AUC": ROCAUC(average="micro", output_transform=lambda x: (x["pred"], to_onehot(x["label"])))
            },
            additional_metrics={
                "Valid_ACC": Accuracy(output_transform=lambda x: (AsDiscrete(threshold_values=True)(x["pred"]), to_onehot(x["label"]))),
                "Valid_Loss": Loss(loss_fn=loss_function, output_transform=lambda x: (x["pred"], x["label"])),
            },
            amp=amp,
            mode=mode
        )

    def run(self, global_epoch: int) -> None:
        
        if global_epoch == 1:
            handlers = [
                StatsHandler(),
                TensorBoardStatsHandler(summary_writer=self.summary_writer),#, output_transform=lambda x: None),
                CheckpointSaver(
                    save_dir=self.output_dir,
                    save_dict={"network": self.network},
                    save_key_metric=True),
                MetricsSaver(save_dir=self.output_dir, metrics=['Valid_AUC', 'Valid_ACC']),
                self.early_stop_handler,
            ]
            self._register_handlers(handlers)

        return super().run(global_epoch=global_epoch)

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super()._iteration(engine, batchdata)
