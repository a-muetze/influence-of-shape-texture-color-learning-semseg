
"""
Custom FCN for Fusion with custom backbone.\
Network input channel number is adaptable.
"""
import torch.nn as nn
import torch
import torchvision.transforms as tvtransf

from models.base_model import BaseModel

from models.deeplabv3resnet_2layer_model import Deeplabv3Resnet2layerModel
from utils.training_utils import find_model_using_name


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FusionModel(BaseModel):
    """ Fusion Model
    """

    def __init__(self, config, in_channels):
        BaseModel.__init__(self, config)
        # define network
        backbone_obj = find_model_using_name(config.deeplab_backbone)

        trafo_list_mean = []
        trafo_list_std = []
        for expert in config.expert_dicts:
            expert_trafo = expert['softmax_transform']
            # softmax should already be in tensor format therefore we skipp ToTensor
            # trafo_list_expertwise[i].append(ToTensor())
            if type(expert_trafo['normalization_mean']) == list:
                trafo_list_mean += expert_trafo['normalization_mean']
                trafo_list_std += expert_trafo['normalization_var']
            else:
                exit("Supported only multiclass with at least 2 classes. Check if softmax output size and normalization parameters correlate")
        self.normalize_softmaxes = tvtransf.Normalize(torch.as_tensor(trafo_list_mean),
                                                      torch.as_tensor(trafo_list_std))
        self.net = FusionBackboneModel(config,
                                       in_channels=in_channels,
                                       backbone_obj=backbone_obj,
                                       softmax_normalization=self.normalize_softmaxes)


class FusionBackboneModel(nn.Module):
    """
    Fusion Mechanismus wird aufgerufen und trainiert
    self.conv1, self.conv2: Convolutional Layer
    self.filter: Input Channels (int)
    self.bn1, self.bn2: Batch Normalization
    return:
    -------
    output: Tensor mit Gewichten
    """

    def __init__(self, config, in_channels, backbone_obj, softmax_normalization):
        super().__init__()

        self.num_experts = len(config.expert_dicts)
        self.log_prob = (config.loss == "NLLLoss")
        self.eps = config.eps

        backbone_cfg = AttrDict({"num_classes": self.num_experts})

        self.backbone = backbone_obj(backbone_cfg, in_channels)
        self.softmax = self.softmax = nn.Softmax(dim=1)

        self.normalize = softmax_normalization
        self.expert_weights = None


    def forward(self, x: torch.Tensor):
        # concatenated softmax from experts
        cat_softmax = x
        # normalized input for improved network training
        normalized_x = self.normalize(torch.clone(x))
        # fusion logits
        expert_weight_logits = self.backbone(normalized_x)['out']
        # fusion softmax
        self.expert_weights = self.softmax(expert_weight_logits)

        # softmaxes per expert
        expert_weights = torch.tensor_split(self.expert_weights,
                                            self.num_experts, dim=1)

        # get original softmaxes of experts per expert
        expert_softmaxes = torch.tensor_split(cat_softmax,
                                              self.num_experts, dim=1) # each split contains as many channels as classes
        if len(expert_softmaxes) != self.num_experts:
            exit("split does not fit!")
        if len(expert_softmaxes) != len(expert_weights):
            exit("split does not fit!")

        # convex combi of softmax of experts and softmax of fusion
        weighted_expert_softmaxes = torch.stack(
            [expert_softmax * expert_weight for expert_softmax, expert_weight in zip(expert_softmaxes, expert_weights)])
        # summing over channels to get a tensor with num_classes channels
        weighted_expert_softmax_cumul = torch.sum(weighted_expert_softmaxes, dim=0)
        # log of cumulated softmax to have adequate input for NLLLoss in pytorch
        weighted_expert_log_softmax_cumul = torch.log(weighted_expert_softmax_cumul + self.eps)

        return weighted_expert_log_softmax_cumul if self.log_prob else weighted_expert_softmax_cumul
