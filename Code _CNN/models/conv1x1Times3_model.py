from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.segmentation.fcn import FCN, FCNHead, IntermediateLayerGetter
from models.base_model import BaseModel

from collections import OrderedDict

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Conv1x1_stack256_256(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(OrderedDict([
            ('conv1', conv1x1(in_channels, 256)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', conv1x1(256,256)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', conv1x1(256,256)),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            # 'conv6', conv1x1(256,256),
            # 'bn6', nn.BatchNorm2d(256),
            # 'relu6', nn.ReLU(inplace=True)
            ])
        )




class Conv1x1Times3Model(BaseModel):
    """ Model with only 1x1 Conv to extract information only based on pixel level, i. e. color
    """
    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)
        # define network
        # backbone_conv1x1 = Conv1x1_stack(in_channels)
        backbone_conv1x1 = Conv1x1_stack256_256(in_channels)
        self.net = _fcn_1x1(
            backbone=backbone_conv1x1,
            num_classes=config.num_classes,
            in_channels=in_channels)


class FCN_1x1_Head(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            conv1x1(in_channels, inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            conv1x1(inter_channels, channels, 1),
        ]

        super().__init__(*layers)



def _fcn_1x1(
    backbone: Conv1x1_stack256_256,
    in_channels:int,
    num_classes: int,
    # aux: Optional[bool],
) -> FCN:
    # return_layers = {"conv1x1_5": "out"}
    return_layers = {"conv3": "out"}
    # if aux:
    #     return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = FCN_1x1_Head(256, num_classes)
    return FCN(backbone, classifier)#, aux_classifier)
