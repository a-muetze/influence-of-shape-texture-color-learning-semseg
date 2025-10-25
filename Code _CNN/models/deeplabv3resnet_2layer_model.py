"""
DeeplabV3 with a 2 layer ResNet backbone
"""
import torch.nn as nn
from torch import Tensor
from torch import flatten
from torchvision.models import resnet
from torchvision.models.resnet import conv3x3, conv1x1
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter
from typing import Any, Optional, Callable, Type, Union, List
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


class ResNet_2Layer(nn.Module):
    """
     Main shortened ResNet class where layer 3 and 4 of the original ResNet18 were removed
    """
    def __init__(
        self,
        block: Type[Union[BasicBlockWithDilation, resnet.Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlockWithDilation) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlockWithDilation, resnet.Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_2layer(
    block: Type[Union[BasicBlockWithDilation, resnet.Bottleneck]],
    layers: List[int],
    weights,
    progress: bool,
    **kwargs: Any,
) -> ResNet_2Layer:
    # if weights is not None:
    #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_2Layer(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet_2layer(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet_2Layer:
    """Reduced ResNet-18 with only 2 ResNet Block each of size 1 \
        original ResNet-18 is from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`.
        Code adapted from torchvision.models.resnet (torchvison version 0.13.0+cu116)
    Args:
       weights: Limited support right now as no error handling is done.
            By default, no pre-trained weights are used. If used, weights need to support
            a get_state_dict(progress) function, which returns the weights as a state dict
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    #weights = resnet.ResNet18_Weights.verify(weights)

    return _resnet_2layer(BasicBlockWithDilation, [2, 2], weights, progress, **kwargs)


def _deeplabv3_resnet_2layer(
    backbone: ResNet_2Layer,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    """ internal representation of DeepLabV3 model with shortened ResNet10 backbone
        Head input channel size is adapted
    """
    return_layers = {"layer2": "out"}
    if aux:
        return_layers["layer1"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(64, num_classes) if aux else None
    classifier = DeepLabHead(128, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def deeplabv3_resnet_2layer(
    *,
    weights,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone = None,
    **kwargs: Any,
) ->DeepLabV3:
    """Constructs a DeepLabV3 model with shortened ResNet10 backbone.
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
        weights_backbone: The pretrained weights for the backbone. Limited support right now as no error handling is done.
            By default, no pre-trained weights are used. If used, weights need to support
            a get_state_dict(progress) function, which returns the weights as a state dict
        **kwargs: unused
    """

    backbone = resnet_2layer(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet_2layer(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class Deeplabv3Resnet2layerModel(BaseModel):
    """ Deeplabv3 with shortened ResNet10 backbone with adaptable input channel number
    """
    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)
        # define network
        self.net = deeplabv3_resnet_2layer(
            weights=None, num_classes=config.num_classes, weights_backbone=None)

        self.net.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)



