'''
Holistic edge detection ([1]). Implementation taken from BrAD ([2]). They use nearly the original version of [3]
    [1] @inproceedings{Xie_ICCV_2015,
         author = {Saining Xie and Zhuowen Tu},
         title = {Holistically-Nested Edge Detection},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2015}
     }
    [2] @InProceedings{Harary_2022_CVPR,
        author    = {Harary, Sivan and Schwartz, Eli and Arbelle, Assaf and Staar, Peter and Abu-Hussein, Shady and Amrani, Elad and Herzig, Roei and Alfassy, Amit and Giryes, Raja and Kuehne, Hilde and Katabi, Dina and Saenko, Kate and Feris, Rogerio S. and Karlinsky, Leonid},
        title     = {Unsupervised Domain Generalization by Learning a Bridge Across Domains},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {5280-5290}
    }
    [3] @misc{pytorch-hed,
         author = {Simon Niklaus},
         title = {A Reimplementation of {HED} Using {PyTorch}},
         year = {2018},
         howpublished = {url{https://github.com/sniklaus/pytorch-hed}}
    }
'''

import numpy as np
import torch
import PIL.Image
from collections import OrderedDict

import torchvision.transforms.functional as fn  # added for visualization reasons

# global variable for holding a persistent single copy of a model
netNetwork = None


class Network(torch.nn.Module):
    def __init__(self, pretrained_hed=True):
        super(Network, self).__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        dl_url = 'http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch'
        if pretrained_hed:
            self.load_state_dict(OrderedDict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                              torch.hub.load_state_dict_from_url(url=dl_url).items()}))

    def forward(self, tenInput):
        tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tenInput = torch.cat([ tenBlue, tenGreen, tenRed ], 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))

@torch.no_grad()
def hed(tenInput):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()

    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    edges = netNetwork(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :]
    return edges.clamp(0.0, 1.0)





if __name__ == "__main__":


    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="/home/datasets/Cityscapes/leftImg8bit/train/", help="/path/to/image/folder/")
    parser.add_argument("--out_folder", default="/home/xxx/data/cityscapes_hed_black_edges/", type=str, help="/path/to/output/folder")
    args = parser.parse_args()

    for root, _, files in os.walk(args.image_folder):
        if files:
            out_dir = os.path.join(args.out_folder, root.split('/')[-2], root.split('/')[-1])
            os.makedirs(out_dir, exist_ok=True)

        for file in files:
            tenInput = torch.FloatTensor(np.ascontiguousarray(
                np.array(PIL.Image.open(os.path.join(root, file))
                         )[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
                )).cuda()

            tenOutput = hed(tenInput).cpu()
            tenOutput = fn.invert(tenOutput)
            PIL.Image.fromarray(
                (tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)
                ).save(os.path.join(out_dir, file))
        print(f"processed: {root.split('/')[-1]}")
