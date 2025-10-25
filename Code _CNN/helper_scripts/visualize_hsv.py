import os
import torch
from PIL import Image
import torchvision.transforms as tvtransf
import torchvision.utils as tvutils
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np



# taken and adapted from https://github.com/limacv/RGB_HSV_HSL
def rgb2hs100v_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = torch.ones_like(rgb[:, 0:1, :, :])
    return torch.cat([hsv_h, hsv_s,hsv_v], dim=1)

# taken and adapted from https://github.com/limacv/RGB_HSV_HSL
def rgb2hsWeightedV_torch(rgb: torch.Tensor, v) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = torch.ones_like(rgb[:, 0:1, :, :])*v
    return torch.cat([hsv_h, hsv_s,hsv_v], dim=1)



def rgb2h100s100v_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_s = torch.ones_like(rgb[:, 0:1, :, :])
    hsv_v = torch.ones_like(rgb[:, 0:1, :, :])
    return torch.cat([hsv_h, hsv_s,hsv_v], dim=1)


# taken and adapted from https://github.com/limacv/RGB_HSV_HSL
def rgb2v_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    hsv_v = cmax
    return hsv_v


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


rootfolder = "tmp"
name = "anisodiff"
for root,_, files in os.walk(os.path.join(rootfolder, name)):
    if root == "results":
        continue
    for file in files:
        img = Image.open(os.path.join(root,file)).convert('RGB')

        tensor_image = tvtransf.ToTensor()(img)

        v = rgb2v_torch(tensor_image.unsqueeze(0))
        rgb_v = torch.concat([v,v,v], dim=1)

        # hs = rgb2hs100v_torch(tensor_image.unsqueeze(0))
        hs = rgb2hsWeightedV_torch(tensor_image.unsqueeze(0),0.5)

        h = rgb2h100s100v_torch(tensor_image.unsqueeze(0))


        remapped_img_hs = hsv2rgb_torch(hs)
        remapped_img_h = hsv2rgb_torch(h)

        tvutils.save_image(rgb_v, os.path.join(rootfolder, f"resultsV50_{name}", f"{file.removesuffix('.png')}_v.png"))
        tvutils.save_image(remapped_img_hs, os.path.join(rootfolder, f"resultsV50_{name}", f"{file.removesuffix('.png')}_hs.png"))



