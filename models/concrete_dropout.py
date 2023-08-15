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
            concrete_dropout=True,
        )
    def forward(self, x: torch.Tensor, member_id=None) -> torch.Tensor:
        x = self.model(x)
        return x

# https://github.com/Alfo5123/ConcreteDropout/blob/master/experiments/uci/uci.ipynb
def heteroscedastic_loss(self, true, mean):
    precision = torch.exp(self.log_prec)

    return torch.sum(precision * (true - mean)**2 - self.log_prec)/true.shape[0]
    
# https://github.com/yaringal/ConcreteDropout/blob/master/spatial-concrete-dropout-keras.ipynb
def heteroscedastic_loss(true, pred):
    mean = pred[:, :D]
    log_var = pred[:, D:]
    precision = K.exp(-log_var)
    return K.sum(precision * (true - mean)**2. + log_var, -1)