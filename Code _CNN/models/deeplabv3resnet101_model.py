import torch.nn as nn
import torchvision
from models.base_model import BaseModel


class Deeplabv3Resnet101Model(BaseModel):
    """ Deeplabv3 with ResNet101 backbone with adaptable input channel number
    """

    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)

        # define network
        self.net = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=None, num_classes=config.num_classes, weights_backbone=None)
        # adaptable input channel size
        self.net.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
