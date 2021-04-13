from typing import Dict, Iterable, Union, Optional
import os
from glob import glob

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
        load_dir: str,
        out_dir: str,
        non_blocking: bool = False,
        post_transform: Optional[Transform] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
    ) -> None:
        self.load_dir = load_dir
        self.out_dir = out_dir

        super().__init__(
            device,
            test_data_loader,
            network,
            non_blocking=non_blocking,
            post_transform=post_transform,
            key_val_metric={
                "Test_AUC": ROCAUC(output_transform=lambda x: (x["pred"], x["label"]))
            },
            additional_metrics={
                "Test_Accuracy": Accuracy(output_transform=lambda x:(AsDiscrete(threshold_values=True)(x["pred"]), x["label"]))
            },
            amp=amp,
            mode=mode
        )

        load_path = glob(os.path.join(self.load_dir, 'network_key_metric*'))[0]
        handlers = [
            StatsHandler(output_transform=lambda x: None),
            CheckpointLoader(load_path=load_path, load_dict={"network": self.network}),
        ]
        self._register_handlers(handlers)

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return super()._iteration(engine, batchdata)

    def run(self) -> None:
        
        testing_dir = os.path.join(self.out_dir, 'testing')
        if not os.path.exists(testing_dir):
            os.mkdir(testing_dir)
        basename = os.path.basename(self.load_dir)
        self.output_dir = os.path.join(testing_dir, basename)
        self._register_handlers(
            [MetricsSaver(save_dir=self.output_dir, metrics=['Test_AUC', 'Test_ACC'])]
        )

        super().run()
