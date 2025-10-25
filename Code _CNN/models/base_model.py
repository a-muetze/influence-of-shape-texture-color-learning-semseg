import torch.nn as nn
import torchvision
from utils.model_utils import weight_init_uniform, weight_reset


class BaseModel(nn.Module):

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.net = None

    def forward(self, x):
        return self.net.forward(x)

    def reset_weights(self):
        # reset all weights as backbone is pretrained by default
        self.net.apply(weight_reset)

    def init_weigths(self):
        self.net.apply(weight_init_uniform)
