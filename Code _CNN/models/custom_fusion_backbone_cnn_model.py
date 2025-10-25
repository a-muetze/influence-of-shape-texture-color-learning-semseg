
"""
Custom FCN for Fusion with custom backbone.\
Network input channel number is adaptable.
"""
import torch.nn as nn

from models.base_model import BaseModel


class CustomFusionBackboneCNNModel(BaseModel):
    """ Deeplabv3 with ResNet18 backbone with adaptable input channel number
    """

    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)
        # define network

        self.net = _CustomFusionBackboneCNNModel(config, in_channels)


class _CustomFusionBackboneCNNModel(nn.Module):
    """
    Fusion Mechanismus wird aufgerufen und trainiert
    self.conv1, self.conv2: Convolutional Layer
    self.filter: Input Channels (int)
    self.bn1, self.bn2: Batch Normalization
    return:
    -------
    output: Tensor mit Gewichten
    """

    def __init__(self, config, in_channels):
        super().__init__()
        self.num_experts = config.num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer3up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer2up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.clf = nn.Conv2d(128, self.num_experts, (1, 1), bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer3up(x)
        x = self.layer2up(x)
        expert_weight_logits = self.clf(x)


        return {"out": expert_weight_logits}
