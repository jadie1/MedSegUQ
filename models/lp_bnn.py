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
            lp_bnn=True,
        )
        self.num_members = args.num_members
        self.device = args.device
    def forward(self, x: torch.Tensor, member_id=None) -> torch.Tensor:
        if self.training:
            x = self.model(x)
            return x
        else:
            batch_size = x.shape[0]                                      # Input shape: [batch_size, in_channels, x, y, z]
            x = torch.cat([x for i in range(self.num_members)], dim=0)   # Make a copy for each member [member*batch_size, in_channels, x, y, z]
            x = self.model(x)                                            # Predict [member*batch_size, out_channels, x, y, z]
            x = x.reshape((self.num_members, batch_size,) + x.shape[1:]) # Reshape [member, batch_size, out_channels, x, y, z]
            pred = x.mean(0) if member_id == None else x[member_id]      # Output shape: [batch_size, channels, x, y, z]
            return pred                                       
