import torch
import torch.nn as nn
import monai.networks.nets as nets

class ModelCT(nets.DenseNet):

    def __init__(self):
        super().__init__(spatial_dims=3, in_channels=1, out_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
