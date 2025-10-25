"""
Fully Convolutional Networks for Semantic Segmentation with Resnet50 backbone.\
    Network input channel number is adaptable.
"""
import torch.nn as nn
import torchvision
from models.base_model import BaseModel


class FCNResnet50Model(BaseModel):
    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)
        # define network
        self.net = torchvision.models.segmentation.fcn_resnet50(
            weights=None, num_classes=config.num_classes, weights_backbone=None)
        # adaptable input channel size
        self.net.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
