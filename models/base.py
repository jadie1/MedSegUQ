import torch
from models.unet import UNet
from monai.networks.layers import Norm

class Model(UNet):
    def __init__(self, args):
        super().__init__(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    def forward(self, x: torch.Tensor, member_id=None) -> torch.Tensor:
        x = self.model(x)
        return x
