import torch.nn as nn
import torch

class FusionAlexNetModel(nn.Module):
    """
    Fusion Mechanismus wird aufgerufen und trainiert
    self.conv1, self.conv2: Convolutional Layer
    self.filter: Input Channels (int)
    self.bn1, self.bn2: Batch Normalization
    return:
    -------
    output: Tensor mit Gewichten
    """

    def __init__(self, config, in_channels, log_probs=True):
        super().__init__()

        self.num_experts=len(config.expert_dicts)
        self.expert_weights = None
        self.log_probs = log_probs

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        # self.layer4up = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU())

        self.layer3up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())

        self.layer2up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.clf = nn.Conv2d(256, self.num_experts, (1, 1), bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        cat_softmax = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer4up(x)
        x = self.layer3up(x)
        x = self.layer2up(x)
        expert_weight_logits = self.clf(x)
        self.expert_weights = self.softmax(expert_weight_logits)

        expert_weights = torch.tensor_split(self.expert_weights,
                                            self.num_experts, dim=1) # each split contains the as many channels as classes


        expert_softmaxes = torch.tensor_split(cat_softmax, self.num_experts, dim=1)
        if len(expert_softmaxes) != self.num_experts:
            exit("split does not fit!")
        if len(expert_softmaxes) != len(expert_weights):
            exit("split does not fit!")

        weighted_expert_softmaxes = torch.stack([expert_softmax * expert_weight
                                                 for expert_softmax, expert_weight
                                                 in zip (expert_softmaxes, expert_weights)])
        weighted_expert_softmax_cumul = torch.sum(weighted_expert_softmaxes, dim= 0)
        weighted_expert_log_softmax_cumul = torch.log(weighted_expert_softmax_cumul)


        return weighted_expert_log_softmax_cumul if self.log_prob else weighted_expert_softmax_cumul
