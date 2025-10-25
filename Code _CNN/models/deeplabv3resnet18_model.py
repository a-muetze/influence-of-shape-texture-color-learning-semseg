"""
DeeplabV3 with reduced backbone (ResNet18)
"""
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet
from torchvision.models.resnet import ResNet, conv3x3
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter
from typing import Any, Optional, Callable
from torchvision.models.segmentation.fcn import FCNHead

from models.base_model import BaseModel


class BasicBlockWithDilation(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet18(*, weights: Optional[resnet.ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
        Adapted due to dilation handling

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = resnet.ResNet18_Weights.verify(weights)

    return resnet._resnet(BasicBlockWithDilation, [2, 2, 2, 2], weights, progress, **kwargs)


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    """ internal representation of Deeplabv3 with ResNet18 backbone
        Head input channel size is adapted
    """
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(256, num_classes) if aux else None
    # adapted DeepLabHead input channel size
    classifier = DeepLabHead(512, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def deeplabv3_resnet18(
    *,
    weights=None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[resnet.ResNet18_Weights] = resnet.ResNet18_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-18 backbone.
        Adapted from torchvision.models.segmentation.deeplabv3 (version 0.13.0+cu116)

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights: Limited support right now as no error handling is done.
            By default, no pre-trained weights are used. If used, weights need to support
            a get_state_dict(progress) function, which returns the weights as a state dict
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet18_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused
    """
    weights_backbone = resnet.ResNet18_Weights.verify(weights_backbone)

    # if weights is not None:
    #     weights_backbone = None
    #     num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    #     aux_loss = _ovewrite_value_param(aux_loss, True)
    # elif num_classes is None:
    #     num_classes = 21

    backbone = resnet18(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class Deeplabv3Resnet18Model(BaseModel):
    """ Deeplabv3 with ResNet18 backbone with adaptable input channel number
    """

    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)
        # define network

        self.net = deeplabv3_resnet18(
            weights=None, num_classes=config.num_classes, weights_backbone=None)

        self.net.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
