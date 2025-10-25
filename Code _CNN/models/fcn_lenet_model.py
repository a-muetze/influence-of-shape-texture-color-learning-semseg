"""
Fully Convolutional Networks for Semantic Segmentation with LeNet backbone.\
    Network input channel number is adaptable.
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torchvision.models._utils import IntermediateLayerGetter
from typing import Any, Optional, Callable

from models.base_model import BaseModel


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6,
                            kernel_size = 5, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,
                            kernel_size = 5, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120,
                            kernel_size = 5, stride = 1, padding = 0)
        # self.linear1 = nn.Linear(120, 84)
        # self.linear2 = nn.Linear(84, 10)
        # self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = nn.ReLU(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = nn.ReLU(x)

        # x = x.reshape(x.shape[0], -1)
        # x = self.linear1(x)
        # x = self.tanh(x)
        # x = self.linear2(x)
        return x


def _fcn_lenet(
    backbone: LeNet,
    num_classes: int,
) -> FCN:
    """ internal representation of FCN model with ResNet18/10 backbone
        Head input channel size is adapted
    """
    return_layers = {"conv3": "out"}

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = FCNHead(120, num_classes)
    return FCN(backbone, classifier)




def fcn_lenet(
    weights = None,
    num_classes: Optional[int] = None,
) -> FCN:
    """Fully-Convolutional Network model with ResNet-18 backbone. Implementation based on the fcn_resnet-50 model
        from torchvision (version 0.13.0+cu116)

    .. betastatus:: segmentation module

    Args:
        weights: Limited support right now as no error handling is done.
            By default, no pre-trained weights are used. If used, weights need to support
            a get_state_dict(progress) function, which returns the weights as a state dict
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet18_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.
    """

    backbone = LeNet()
    model = _fcn_lenet(backbone, num_classes)

    if weights is not None:
        model.load_state_dict(weights)

    return model


class FcnLeNetModel(BaseModel):
    """Fully-Convolutional Network model with a LeNet backbone. \
        Code adapted from torchvision.models.segmentation (torchvison version 0.13.0+cu116). \
        Basic implementation from the `Fully Convolutional \
        Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper."""
    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)
        # define network
        self.net = fcn_lenet(weights=None, num_classes=config.num_classes)
        # adaptable input channel size
        self.net.backbone.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0, bias=True)
