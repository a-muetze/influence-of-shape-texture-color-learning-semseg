'''This file is taken from https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
and was extended by the marked classes
'''

import numpy as np
from PIL import Image, ImageOps
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

import numbers


# adapted to allow non squared crops
def pad_if_smaller(img, size, fill=0):
    ow, oh = img.size()[-2:]
    cw, ch = size
    padw = cw - ow if ow < cw else 0
    padh = ch - oh if oh < ch else 0
    img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Compose3(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, hed):
        for t in self.transforms:
            image, target, hed = t(image, target, hed)
        return image, target, hed


class RandomResize(object):
    ''' randomly resizes images within the given range based on the shorter edge of the image. Aspectratio is preserved '''
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

# Added class
class Resize(object):
    ''' resize images to specific size, aspect ratio will be ignored'''
    def __init__(self, width, height=None):
        self.width = width
        if height is None:
            height = width
        self.height = height

    def __call__(self, image, target):
        size = [self.width, self.height]
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


# class has been modified
class RandomCropTensor(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomCrop(object):
    """Returns an image of size 'size' that is a random crop of the original.

    Args:
        size: Size of the croped image.
        padding: Number of pixels to be placed around the original image.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        """Returns randomly cropped image."""
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), T.InterpolationMode.NEAREST),
                    mask.resize((tw, th), T.InterpolationMode.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)))

class RandomCrop3(object):
    """Returns an image of size 'size' that is a random crop of the original.

    Args:
        size: Size of the croped image.
        padding: Number of pixels to be placed around the original image.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, img2):
        """Returns randomly cropped image."""
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            img2 = ImageOps.expand(img2, border=self.padding, fill=0)

        assert img.size == mask.size and mask.size == img2.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask, img2
        if w < tw or h < th:
            return (img.resize((tw, th), T.InterpolationMode.NEAREST),
                    mask.resize((tw, th), T.InterpolationMode.NEAREST),
                    img2.resize((tw, th), T.InterpolationMode.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                img2.crop((x1, y1, x1 + tw, y1 + th)))


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ToTensor3(object):
    def __call__(self, image, target, img2):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        img2 = F.to_tensor(img2)
        return image, target, img2


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Normalize3(object):
    def __init__(self, mean, std, mean2, std2):
        self.mean = mean
        self.std = std
        self.mean2 = mean2
        self.std2 = std2

    def __call__(self, image, target, img2):
        image = F.normalize(image, mean=self.mean, std=self.std)
        img2 = F.normalize(img2, mean=self.mean2, std=self.std2)
        return image, target, img2


# # taken from Guillermo Gómez guillermogotre  (https://gist.github.com/guillermogotre/844024ac37d35c8a00a6133887cbd18b)
# class UnNormalize(torchvision.transforms.Normalize):
#     def __init__(self,mean,std,*args,**kwargs):
#         new_mean = [-m/s for m,s in zip(mean,std)]
#         new_std = [1/s for s in std]
#         super().__init__(new_mean, new_std, *args, **kwargs)

# # imagenet_norm = dict(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# # UnNormalize(**imagenet_norm)

## taken from https://github.com/pytorch/vision/issues/848#issuecomment-482547380
# class Denormalize(object):
#     def __init__(self, mean, std, inplace=False):
#         self.mean = mean
#         self.demean = [-m/s for m, s in zip(mean, std)]
#         self.std = std
#         self.destd = [1/s for s in std]
#         self.inplace = inplace

#     def __call__(self, tensor):
#         tensor = F.normalize(tensor, self.demean, self.destd, self.inplace)
#         # clamp to get rid of numerical errors
#         return torch.clamp(tensor, 0.0, 1.0)

#################################################################
# For Pascal Context
#################################################################

'''
Code source: https://github.com/TornikeAm/Semantic-Segmentation-on-Pascal-VOC/blob/main/Semantic%20Segmentation.ipynb
'''
class Crop():
    '''
    Code source : https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/datasets/base.py
    '''
    def __init__(self, base_size=520, crop_size=480, fill_img_value=0, fill_mask_value=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill_img_value = fill_img_value
        self.fill_mask_value = fill_mask_value

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        #print(crop_size)
        # random scale (short edge)
        w, h = img.size
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=self.fill_img_value)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill_mask_value)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        #print(np.shape(img))
        # final transform
        return img, mask



