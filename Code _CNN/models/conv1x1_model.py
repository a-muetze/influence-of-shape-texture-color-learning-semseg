from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.segmentation.fcn import FCN, FCNHead, IntermediateLayerGetter
from models.base_model import BaseModel


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Conv1x1_stack256_256(nn.Module):
    def __init__(self, in_channels):
        super(Conv1x1_stack256_256, self).__init__()
        self.conv1x1_1 = conv1x1(in_channels, 256)
        self.bn = nn.BatchNorm2d(256)
        self.conv1x1_2 = conv1x1(256,256)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv1x1_3 = conv1x1(256,256)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv1x1_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1x1_2(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.conv1x1_3(x)
        return x



# class Conv1x1_stack(nn.Module):
#     def __init__(self, in_channels):
#         super(Conv1x1_stack, self).__init__()
#         self.conv1x1_1 = conv1x1(in_channels, 32)
#         self.conv1x1_2 = conv1x1(32, 64)
#         self.conv1x1_3 = conv1x1(64, 128)
#         self.conv1x1_4 = conv1x1(128, 256)
#         self.conv1x1_5 = conv1x1(256, 128)

#     def forward(self, x):
#         x = F.relu(self.conv1x1_1(x))
#         x = F.relu(self.conv1x1_2(x))
#         x = F.relu(self.conv1x1_3(x))
#         x = F.relu(self.conv1x1_4(x))
#         x = F.relu(self.conv1x1_5(x))
#         return x

class Conv1x1Model(BaseModel):
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
    return_layers = {"conv1x1_2": "out"}
    # if aux:
    #     return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # aux_classifier = FCNHead(1024, num_classes) if aux else None
    # classifier = FCNHead(128, num_classes)
    classifier = FCN_1x1_Head(256, num_classes)
    return FCN(backbone, classifier)#, aux_classifier)
