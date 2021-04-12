from typing import Dict, Iterable, Union, Optional

import torch
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.metrics import Accuracy
from monai.engines import SupervisedEvaluator
from monai.transforms import Transform, AsDiscrete
from monai.handlers import CheckpointLoader, StatsHandler, MetricsSaver, ROCAUC
from monai.utils import ForwardMode

class Tester(SupervisedEvaluator):

    def __init__(
        self,
        device: torch.device,
        test_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        load_path: str,
        non_blocking: bool = False,
        post_transform: Optional[Transform] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
    ) -> None:
        super().__init__(
            device,
            test_data_loader,
            network,
            non_blocking=non_blocking,
            post_transform=post_transform,
            key_val_metric={
                "Valid_AUC": ROCAUC(output_transform=lambda x: (x["pred"], x["label"]))
            },
            additional_metrics={
                "Valid_Accuracy": Accuracy(output_transform=lambda x:(AsDiscrete(threshold_values=True)(x["pred"]), x["label"]))
            },
            amp=amp,
            mode=mode
        )

        handlers = [
            StatsHandler(output_transform=lambda x: None),
            CheckpointLoader(load_path=load_path, load_dict={"network": self.network}),
        ]
        self._register_handlers(handlers)

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super()._iteration(engine, batchdata)
