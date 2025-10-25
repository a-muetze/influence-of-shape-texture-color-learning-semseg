import logging
import os
from collections import namedtuple
from typing import Any, Dict, Tuple
import cv2

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as tvtransf
import torch

from helper_scripts.custom_transforms import Normalize, ToTensor, RandomCrop
from helper_scripts.custom_transforms import RandomCropTensor, Compose, Crop

class CityscapesHEDBlackEdges19classes(Dataset):
    """
    Shape data based on Cityscapes (http://www.cityscapes-dataset.com/):
    Labels based on 19 training classes from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    1) holistic edge detection based. Images can be generated with the help of datasets/lib_hed/hed.py. Make sure images are white with black edges
    2) Anisotrophic diffusion based
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]

    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)

    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))

    # HED(c)
    # normalization_mean = 0.9037  # HED black edges (over whole training-dataset)
    # normalization_var = 0.18 # HED black edges (over whole training-dataset))

    # anisodiff RGB(c)
    # normalization_mean = [0.2544, 0.2949, 0.2515] # Anisotrophic diffusion
    # normalization_var = [0.1803, 0.1854, 0.1826] #  Anisotrophic diffusion

    # anisodiff gray (c)
    # normalization_mean = 0.2778 # Anisotrophic diffusion (gray scale)
    # normalization_var = 0.1832 # Anisotrophic diffusion (gray scale)

    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 transform: Dict = None,
                 mode: str = "fine",
                 target_type: str = "labelIds",
                 ignore_cities: list = [], #['dusseldorf', 'ulm'],
                 train_on_train_id: bool = True
                 ) -> None:
        """
        Cityscapes dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.train_on_train_id = train_on_train_id
        self.gray = False

        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCrop(tuple(transform['random_crop'])))
        trafo_list.append(ToTensor())
        if type(transform['normalization_mean']) == list and len(transform['normalization_mean']) > 1:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)

        # data root paths
        self.images_dir = os.path.join(self.root, self.split)
        self.targets_dir = os.path.join(self.target_root, self.mode, self.split.split('_')[0])

        self.images = []
        self.targets = []

        # data paths
        for city in os.listdir(self.images_dir):
            if (self.split == 'train' or self.split == 'val') and city not in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)

                for file_name in os.listdir(img_dir):
                    target_name_prefix = file_name.split("_leftImg8bit")[0]
                    target_name = f'{target_name_prefix}_{self.mode}_{target_type}.png'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))

        if self.split == 'train_testsplit' or self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra':
            img_dir = self.images_dir
            target_dir = self.target_root

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, file_name))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.gray:
            image = Image.open(self.images[index]).convert('L')
            # image = fn.invert(image)
        else:
            image = Image.open(self.images[index]).convert('RGB')

        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        image, target = self.transform(image, target)
        path = self.images[index]

        return image, target, path

    def __len__(self) -> int:
        return len(self.images)




class CityscapesTexture2(Dataset):
    """`
    Texture data based on Cityscapes Dataset http://www.cityscapes-dataset.com/
    Images can be generated with the help of script polygon_semseg.py in the Code_texture-generation folder
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]


    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]

    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)

    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))



    # upsampled_texture_images_in_contour (c)
        # normalization_mean = (0.2753, 0.3099, 0.2740)
        # normalization_var = (0.2081, 0.2134, 0.2091)

    # gray scale upsampled_texture_images_in_contour (c)
        # # normalization_mean [0.2955]
        # normalization_var [0.2102]


    # for gray scale wo contour: (c)
        # normalization_mean(0.2953)
        # normalization_var(0.2114)

    # wo contour fill (c)
        # normalization_mean = (0.2751, 0.3097, 0.2739)
        # normalization_var = (0.2093, 0.2146, 0.2105)
    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 transform: Dict = None,
                 mode: str = "fine",
                 target_type: str = "labelTrainIds",
                 ignore_cities: list = ['dusseldorf', 'ulm', 'hanover'],
                 train_on_train_id: bool = False
                 ) -> None:
        """
        Cityscapes dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.train_on_train_id = train_on_train_id
        self.gray = False

        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCrop(tuple(transform['random_crop'])))
        trafo_list.append(ToTensor())
        if type(transform['normalization_mean']) == list  and len(transform['normalization_mean']) > 1:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)


        self.images = []
        self.targets = []

        # use texture voronoi diagrams
        if (self.split == 'train' or self.split == 'val' or self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra'):
            # data root paths
            self.images_dir = os.path.join(self.root, self.split)
            self.targets_dir = os.path.join(self.target_root, self.split)

            # iterate data paths
            for file_name in os.listdir(self.images_dir):
                target_name_parts = file_name.removesuffix('.png')
                target_name = f'{target_name_parts}_train_id.png'
                self.images.append(os.path.join(self.images_dir, file_name))
                self.targets.append(os.path.join(self.targets_dir, target_name))

        # use a subset of original cityscapes training data
        elif self.split == 'train_testsplit':
            print("This is a subset of the training set! Not an extra split")
            # data root
            self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split.split('_')[0])
            self.targets_dir = os.path.join(self.root, self.mode, self.split.split('_')[0])
            for city in os.listdir(self.images_dir):
                if city in ignore_cities: # counter intuitive wording but describes the citys used as subset
                    img_dir = os.path.join(self.images_dir, city)
                    target_dir = os.path.join(self.targets_dir, city)
                    for file_name in os.listdir(img_dir):
                        target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
                        self.images.append(os.path.join(img_dir, file_name))
                        self.targets.append(os.path.join(target_dir, target_name))
        else:
            exit("Choose one of the splits 'train', ''val' or 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra' \
                or if you know what you are doing 'train_testsplit'")


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.gray:
            image = Image.open(self.images[index]).convert('L')
        else:
            image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        image, target = self.transform(image, target)
        path = self.images[index]

        return image, target, path

    def __len__(self) -> int:
        return len(self.images)




class CityscapesTextureHS(Dataset):
    """`
    Texture data based on Cityscapes Dataset http://www.cityscapes-dataset.com/
    Images can be generated with the help of script polygon_semseg.py in the Code_texture-generation folder
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]


    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]

    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)

    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    #  upsampled_texture_images_in_contour (c)
    """Normalization parameters rgb2hs"""
    # mean = (0.3411435, 0.2125287)
    # std = (0.1262746, 0.1077188)


    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 transform: Dict = None,
                 mode: str = "fine",
                 target_type: str = "labelTrainIds",
                 ignore_cities: list = ['dusseldorf', 'ulm', 'hanover'],
                 train_on_train_id: bool = False
                 ) -> None:
        """
        Texture dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.train_on_train_id = train_on_train_id
        self.gray = False

        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCropTensor(tuple(transform['random_crop'])))
        if type(transform['normalization_mean']) == list and len(transform["normalization_mean"]) > 1:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)


        self.images = []
        self.targets = []

        # use texture voronoi diagrams
        if (self.split == 'train' or self.split == 'val' or self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra'):
            # data root paths
            self.images_dir = os.path.join(self.root, self.split)
            self.targets_dir = os.path.join(self.target_root, self.split)

            # iterate data paths
            for file_name in os.listdir(self.images_dir):
                target_name_parts = file_name.removesuffix('.png')
                target_name = f'{target_name_parts}_train_id.png'
                self.images.append(os.path.join(self.images_dir, file_name))
                self.targets.append(os.path.join(self.targets_dir, target_name))

        # use a subset of original cityscapes training data
        elif self.split == 'train_testsplit':
            print("This is a subset of the training set! Not an extra split")
            # data root
            self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split.split('_')[0])
            self.targets_dir = os.path.join(self.root, self.mode, self.split.split('_')[0])
            for city in os.listdir(self.images_dir):
                if city in ignore_cities: # counter intuitive wording but describes the citys used as subset
                    img_dir = os.path.join(self.images_dir, city)
                    target_dir = os.path.join(self.targets_dir, city)
                    for file_name in os.listdir(img_dir):
                        target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
                        self.images.append(os.path.join(img_dir, file_name))
                        self.targets.append(os.path.join(target_dir, target_name))
        else:
            exit("Choose one of the splits 'train', ''val' or 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra' \
                or if you know what you are doing 'train_testsplit'")


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        tensor_image = tvtransf.ToTensor()(image)

        if self.gray:
            v = self.rgb2v_torch(tensor_image.unsqueeze(0)).squeeze(dim=0)
            tensor_img = v

        else:
            hs = self.rgb2hs_torch(tensor_image.unsqueeze(0)).squeeze()
            tensor_img = hs

        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        tensor_target = torch.as_tensor(np.array(target), dtype=torch.int64)
        tensor_img, target = self.transform(tensor_img, tensor_target)

        path = self.images[index]

        return tensor_img, target, path

    def __len__(self) -> int:
        return len(self.images)


    # taken and adapted from https://github.com/limacv/RGB_HSV_HSL
    def rgb2hs_torch(self, rgb: torch.Tensor) -> torch.Tensor:
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
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s], dim=1)


    # taken and adapted from https://github.com/limacv/RGB_HSV_HSL
    def rgb2v_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        hsv_v = cmax
        return hsv_v




class CityscapesHSV19classes(Dataset):
    """`
    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]


    """Normalization parameters opencv"""
    # mean = (0.2321938, 0.1922899)
    # std = (0.0616693, 0.0793071)

    """Normalization parameters rgb2hs""" #(c)
    # mean = (0.3287114, 0.1922254)
    # std = (0.0873381, 0.0792537)

    """Normalization parameters rgb2v""" #(c)
    # mean = (0.3269421)
    # std = (0.1816357)



    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 transform: Dict = None, # Optional[Callable] = None,
                 mode: str = "fine",
                 target_type: str = "labelTrainIds",
                 ignore_cities: list = [],
                 train_on_train_id: bool = False
                ) -> None:
        """
        Cityscapes dataset loader
        """
        if label_root != img_root:
            print("Are you using the original Cityscapes data and structure? Please verify you are using the right data. Will exit now...")
            exit()
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.transform = transform
        self.train_on_train_id = train_on_train_id
        self.gray = False

        # build transformation
        trafo_list = []
        # trafo_list.append(ToTensor())
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCropTensor(tuple(transform['random_crop'])))
        if type(transform['normalization_mean']) == list and len(transform['normalization_mean']) > 1:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)

        # data root
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split.split('_')[0])
        self.targets_dir = os.path.join(self.root, self.mode, self.split.split('_')[0])

        self.images = []
        self.targets = []

        if self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra':
            # data root paths
            self.images_dir = os.path.join(self.root, self.split, "leftImg8bit")
            self.targets_dir = os.path.join(self.target_root, self.split, "pseudo_gt_fine")

            # iterate data paths
            for file_name in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, file_name))
                self.targets.append(os.path.join(self.targets_dir, file_name))

        for city in os.listdir(self.images_dir):
            if (self.split == 'train' or self.split == 'val') and city not in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))
            # if self.split == 'train_testsplit' and city in ignore_cities:
            #     img_dir = os.path.join(self.images_dir, city)
            #     target_dir = os.path.join(self.targets_dir, city)
            #     for file_name in os.listdir(img_dir):
            #         target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
            #         self.images.append(os.path.join(img_dir, file_name))
            #         self.targets.append(os.path.join(target_dir, target_name))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        tensor_image = tvtransf.ToTensor()(image)
        if self.gray:
            v = self.rgb2v_torch(tensor_image.unsqueeze(0)).squeeze(dim=0)
            tensor_img = v
        else:
            hs = self.rgb2hs_torch(tensor_image.unsqueeze(0)).squeeze()
            tensor_img = hs
        target = Image.open(self.targets[index])
        # hs = hsv[...,0:-1]
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)
        tensor_target = torch.as_tensor(np.array(target), dtype=torch.int64)
        tensor_img, target = self.transform(tensor_img, tensor_target)
        path = self.images[index]

        return tensor_img, target, path

    def __len__(self) -> int:
        return len(self.images)

    # taken and adapted from https://github.com/limacv/RGB_HSV_HSL
    def rgb2hs_torch(self, rgb: torch.Tensor) -> torch.Tensor:
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
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s], dim=1)


    # taken and adapted from https://github.com/limacv/RGB_HSV_HSL
    def rgb2v_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        hsv_v = cmax
        return hsv_v

class CityscapesAnisodiffHS19classes(Dataset):
    """
    Shape data based on Cityscapes (http://www.cityscapes-dataset.com/):
    Labels based on 19 training classes from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    Anisotrophic diffusion based (EED)
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]

    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)

    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))

    # # Anisodiff HS (c)
    # normalization_mean = [0.3285040, 0.2477634]
    # normalization_var = [0.0729014, 0.1185639]


    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 transform: Dict = None,
                 mode: str = "fine",
                 target_type: str = "labelIds",
                 ignore_cities: list = [], #['dusseldorf', 'ulm'],
                 train_on_train_id: bool = True
                 ) -> None:
        """
        Cityscapes dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.train_on_train_id = train_on_train_id
        self.gray = False

        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCropTensor(tuple(transform['random_crop'])))
        if type(transform['normalization_mean']) == list and len(transform['normalization_mean']) > 1:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)

        # data root paths
        self.images_dir = os.path.join(self.root, self.split)
        self.targets_dir = os.path.join(self.target_root, self.mode, self.split.split('_')[0])

        self.images = []
        self.targets = []

        # data paths
        for city in os.listdir(self.images_dir):
            if (self.split == 'train' or self.split == 'val') and city not in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)

                for file_name in os.listdir(img_dir):
                    target_name_prefix = file_name.split("_leftImg8bit")[0]
                    target_name = f'{target_name_prefix}_{self.mode}_{target_type}.png'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))

        if self.split == 'train_testsplit' or self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra':
            img_dir = self.images_dir
            target_dir = self.target_root

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, file_name))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        tensor_image = tvtransf.ToTensor()(image)
        if self.gray:
            v = self.rgb2v_torch(tensor_image.unsqueeze(0)).squeeze(dim=0)
            tensor_img = v
        else:
            hs = self.rgb2hs_torch(tensor_image.unsqueeze(0)).squeeze()
            tensor_img = hs
        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)
        tensor_target = torch.as_tensor(np.array(target), dtype=torch.int64)
        tensor_img, target = self.transform(tensor_img, tensor_target)
        path = self.images[index]

        return tensor_img, target, path

    def __len__(self) -> int:
        return len(self.images)

    # taken and adapted from https://github.com/limacv/RGB_HSV_HSL
    def rgb2hs_torch(self, rgb: torch.Tensor) -> torch.Tensor:
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
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s], dim=1)


    # taken and adapted from https://github.com/limacv/RGB_HSV_HSL
    def rgb2v_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        hsv_v = cmax
        return hsv_v



class CityscapesRgb19classes(Dataset):
    """`
    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]

    """Normalization parameters"""
    # ImageNet normalization
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)

    # Cityscapes training data normalization (c)
    # mean = (0.2868955, 0.3251328, 0.2838913)
    # std = (0.1761364, 0.1809918, 0.1777224)

    # grayscale (c)
    # mean = (0.3090155)
    # std = (0.1786242)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 transform: Dict = None, # Optional[Callable] = None,
                 mode: str = "fine",
                 target_type: str = "labelTrainIds",
                 ignore_cities: list = [],
                 train_on_train_id: bool = True
                ) -> None:
        """
        Cityscapes dataset loader
        """
        if label_root != img_root:
            print("Are you using the original Cityscapes data and structure? Please verify you are using the right data. Will exit now...")
            exit()
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.transform = transform
        self.train_on_train_id = train_on_train_id
        self.gray = False

        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCrop(tuple(transform['random_crop'])))
        trafo_list.append(ToTensor())
        if type(transform['normalization_mean']) == list and len(transform['normalization_mean']) >1:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)

        # data root
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split.split('_')[0])
        self.targets_dir = os.path.join(self.root, self.mode, self.split.split('_')[0])

        self.images = []
        self.targets = []

        if self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra':
            # data root paths
            self.images_dir = os.path.join(self.root, self.split, "leftImg8bit")
            self.targets_dir = os.path.join(self.target_root, self.split, "pseudo_gt_fine")

            # iterate data paths
            for file_name in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, file_name))
                self.targets.append(os.path.join(self.targets_dir, file_name))

        for city in os.listdir(self.images_dir):
            if (self.split == 'train' or self.split == 'val') and city not in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))
            # if self.split == 'train_testsplit' and city in ignore_cities:
            #     img_dir = os.path.join(self.images_dir, city)
            #     target_dir = os.path.join(self.targets_dir, city)
            #     for file_name in os.listdir(img_dir):
            #         target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
            #         self.images.append(os.path.join(img_dir, file_name))
            #         self.targets.append(os.path.join(target_dir, target_name))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.gray:
            image = Image.open(self.images[index]).convert('L')
        else:
            image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        image, target = self.transform(image, target)
        path = self.images[index]

        return image, target, path

    def __len__(self) -> int:
        return len(self.images)


class CityscapesFusionSoftmax19classes(Dataset):
    """
    Softmax data from different experts trained on Cityscapes shape, texture and color (http://www.cityscapes-dataset.com/):
    Labels based on 19 training classes from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    Softmax is generated by main_infer_and_visualize_results.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]

    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)

    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))
    


    def __init__(self,
                 img_root: [str], # list of softmax inputs (misleading name is due to interface reasons)
                 label_root: str,
                 split: str,
                 transform: Dict = None,
                 mode: str = "fine",
                 target_type: str = "labelTrainIds",
                 ignore_cities: list = [], #['dusseldorf', 'ulm'],
                 train_on_train_id: bool = True,
                 expert_dicts: Dict = None
                 ) -> None:
        """
        Cityscapes dataset loader
        """
        self.softmax_root_list = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.train_on_train_id = train_on_train_id

        # Handle mutli expert transformations
        self.trafo_functions_global = None
        # self.trafo_functions_experts = []

        transform_global = transform

        # build transformation
        if self.split == 'train' and 'random_crop' in transform_global:
            self.trafo_functions_global = RandomCropTensor(tuple(transform_global['random_crop']))

        # data root paths
        self.softmax_dir = [os.path.join(softmax_root, self.split) for softmax_root in self.softmax_root_list]
        self.targets_dir = os.path.join(self.target_root, self.mode, self.split.split('_')[0])

        if self.split == "train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra":
            self.targets_dir = os.path.join(self.target_root, self.split, "pseudo_gt_fine")

        self.softmax_list =  [[] for i in range(len(expert_dicts))]
        self.targets = []

        cnt = 0
        for file_name in os.listdir(self.softmax_dir[0]):
            # skipp if it is a directory
            if not file_name[-3:]==".pt":
                continue
            # if cnt >= 50:
            #     break
            # cnt += 1
            # softmax path
            for idx in range(len(self.softmax_dir)):
                self.softmax_list[idx].append(os.path.join(self.softmax_dir[idx], file_name))
            # target path
            if self.split == "train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra":
                target_dir = self.targets_dir
                target_name = file_name.removesuffix('.pt') + ".png"
            else:
                city = file_name.split("_")[0]
                target_dir = os.path.join(self.targets_dir, city)
                target_name_prefix = file_name.split("_leftImg8bit")[0]
                target_name = f'{target_name_prefix}_{self.mode}_{target_type}.png'
            self.targets.append(os.path.join(target_dir, target_name))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        softmax_list = [torch.load(expert_softmax_list[index]).to(torch.float32) for expert_softmax_list in self.softmax_list] # map_location=torch.device('cpu')

        cat_softmax = torch.concat(softmax_list)

        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        if self.trafo_functions_global:
            cat_softmax, target = self.trafo_functions_global(cat_softmax, target)
        paths = [exp_path[index] for exp_path in self.softmax_list]

        return cat_softmax, target, paths

    def __len__(self) -> int:
        return len(self.softmax_list[0])




class Softmax19classes(Dataset):
    def __init__(self,
                 softmax_root: str,
                 label_root: str,
                 split: str,
                 mode: str = "fine",
                 transform: Dict = None,
                 target_type: str = "labelTrainIds",
                 train_on_train_id: bool = False):
        self.root = softmax_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.gray = False
        self.train_on_train_id = train_on_train_id

        self.softmax = []
        self.targets = []

        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCrop(tuple(transform['random_crop'])))
        # trafo_list.append(ToTensor())
        if type(transform['normalization_mean']) == list:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)

        if (self.split == 'train' or self.split == 'val' or self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra'):
            # data root paths
            self.softmax_dir = os.path.join(self.root, self.split)
            self.targets_dir = os.path.join(self.target_root, self.mode, self.split.split('_')[0])

            # iterate data paths
            for file_name in os.listdir(self.softmax_dir):
                # skipp if it is a directory
                if not file_name[-3:]==".pt":
                    continue

                # target path
                city = file_name.split("_")[0]
                target_dir = os.path.join(self.targets_dir, city)
                target_name_prefix = file_name.split("_leftImg8bit")[0]
                target_name = f'{target_name_prefix}_{self.mode}_{target_type}.png'
                self.targets.append(os.path.join(target_dir, target_name))

                self.softmax.append(os.path.join(self.softmax_dir, file_name))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        softmax_tensor = torch.load(self.softmax[index], map_location=torch.device('cpu'))
        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        softmax_tensor, target = self.transform(softmax_tensor, target)
        path = self.softmax[index]

        return softmax_tensor, target, path

    def __len__(self) -> int:
        return len(self.softmax)



class CityscapesRandomTextureInContour19classes(Dataset):
    """
    Shape data based on Cityscapes (http://www.cityscapes-dataset.com/):
    Labels based on 19 training classes from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    1) texture was merged into one mosaic image.
    2) segments are filled randomly with texture patches
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]

    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)

    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 transform: Dict = None,
                 mode: str = "fine",
                 target_type: str = "labelIds",
                 ignore_cities: list = [], #['dusseldorf', 'ulm'],
                 train_on_train_id: bool = True
                 ) -> None:
        """
        Cityscapes dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.train_on_train_id = train_on_train_id
        self.gray = False

        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCrop(tuple(transform['random_crop'])))
        trafo_list.append(ToTensor())
        if type(transform['normalization_mean']) == list and len(transform['normalization_mean']) > 1:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)

        # data root paths
        self.images_dir = os.path.join(self.root, self.split)
        self.targets_dir = os.path.join(self.target_root, self.split)

        self.images = []
        self.targets = []

        # data paths
        for city in os.listdir(self.images_dir):
            # if (self.split == 'train' or self.split == 'val') and city not in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)

                for file_name in os.listdir(img_dir):
                    target_name_postfix = file_name.split("random_texture_in_contour_")[-1]
                    target_name = f'train_id_{target_name_postfix}'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))

        # if self.split == 'train_testsplit' or self.split == 'train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra':
        #     img_dir = self.images_dir
        #     target_dir = self.target_root

        #     for file_name in os.listdir(img_dir):
        #         self.images.append(os.path.join(img_dir, file_name))
        #         self.targets.append(os.path.join(target_dir, file_name))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.gray:
            image = Image.open(self.images[index]).convert('L')
            # image = fn.invert(image)
        else:
            image = Image.open(self.images[index]).convert('RGB')

        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.train_on_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        image, target = self.transform(image, target)
        path = self.images[index]

        return image, target, path

    def __len__(self) -> int:
        return len(self.images)





################################################################################
##########################  Pascal Context Dataloader  #########################
################################################################################


class PascalContext33classes(Dataset):
    PascalContextClass = namedtuple('PascalContextClass', ['name',
                                                           'id', 
                                                           'train_id',
                                                           'color'
                                                           ])
    labels = [
        PascalContextClass('void',          0,  255,(  0,  0,  0)),
        PascalContextClass('aeroplane',     2,    0,(128,  0,  0)),
        PascalContextClass('bicycle',      23,    1,(  0,128,  0)),
        PascalContextClass('bird',         25,    2,(128,128,  0)),
        PascalContextClass('boat',         31,    3,(  0,  0,128) ),
        PascalContextClass('bottle',       34,    4,(128,  0,128)),
        PascalContextClass('bus',          45,    5,(  0,128,128)),
        PascalContextClass('car',          59,    6,(128,128,128)),
        PascalContextClass('cat',          65,    7,( 64,  0,  0)),
        PascalContextClass('chair',        72,    8,(192,  0,  0)),
        PascalContextClass('cow',          98,    9,( 64,128,  0)),
        PascalContextClass('diningtabel', 397,   10,(192,128,  0)),
        PascalContextClass('dog',         113,   11,( 64,  0,128)),
        PascalContextClass('horse',       207,   12,(192,  0,128)),
        PascalContextClass('motorbike',   258,   13,( 64,128,128)),
        PascalContextClass('person',      284,   14,(192,128,128)),
        PascalContextClass('pottedplant', 308,   15,(  0, 64,  0)),
        PascalContextClass('sheep',       347,   16,(128, 64,  0)),
        PascalContextClass('sofa',        368,   17,(  0,192,  0)),
        PascalContextClass('train',       416,   18,(128,192,  0)),
        PascalContextClass('tvmonitor',   427,   19,(  0, 64,128)),
        PascalContextClass('sky',         360,   20,(  0,192, 64)),
        PascalContextClass('grass',       187,   21,(  0,  0,192)),
        PascalContextClass('ground',      189,   22,(128,  0,192)),
        PascalContextClass('road',        324,   23,( 64,128,192)),
        PascalContextClass('building',     44,   24,(192, 64,  0)),
        PascalContextClass('tree',        420,   25,( 64,192,  0)),
        PascalContextClass('water',       445,   26,(192, 64, 64)),
        PascalContextClass('mountain',    259,   27,( 64,  0, 64)),
        PascalContextClass('wall',        440,   28,(128, 64,128)),
        PascalContextClass('floor',       158,   29,(128,  0, 64)),
        PascalContextClass('track',       415,   30,(128, 64,192)),
        PascalContextClass('keyboard',    220,   31,(  0,128,192)),
        PascalContextClass('ceiling',      68,   32,(192,192,  0))
    ]
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
  
    # for i in range(len(labels)):
    #     if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
    #         ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        #if labels[i].train_id not in ignore_in_eval_ids:
        train_ids.append(labels[i].train_id)
        color_palette_train_ids[labels[i].train_id] = labels[i].color
        train_id2id.append(labels[i].id)
       
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_2label = {label.color: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    # mean: [  ]
    # std: [  ] 

    def __init__(self,
                    img_root: str,
                    label_root: str,
                    split: str,
                    transform: Dict = None,
                    mode: str = "gtFine",
                    #target_type: str = "labelTrainIDs",
                    train_on_train_id: bool = True,
                    gray: bool=False,
                    rgb:bool=True
                    ) -> None:
        """
        Pascal Context dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' 
        self.train_on_train_id = train_on_train_id
        self.gray = gray
        self.rgb = rgb
        self.hsv = False
        if not self.gray and not self.rgb:
            self.hsv = True
        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCrop(tuple(transform['random_crop'])))
        
        
        trafo_list.append(ToTensor())
        if type(transform['normalization_mean']) == list:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
        else:
            trafo_list.append(Normalize(transform['normalization_mean'],
                                        transform['normalization_var']))
            self.gray = True
        self.transform = Compose(trafo_list)

        # data root paths
        if self.split == 'val_train':
            exit("Please use the val split since a seperate test split exists")
            self.split = 'train'
        self.images_dir = os.path.join(self.root)
        self.targets_dir = os.path.join(self.target_root)

        self.images = []
        self.targets = []

        # data paths
        # for sequence in os.listdir(self.images_dir):
        #     if (self.split == 'train' or self.split == 'val') and sequence not in ignore_sequences:
        #         img_dir = os.path.join(self.images_dir, sequence)
        #         target_dir = os.path.join(self.targets_dir, sequence)
        for root, dirs, files in os.walk(os.path.join(self.targets_dir, self.split)):
            for file in files:
                img_name = f'{file.split(".")[0]}.jpg'
                
                self.images.append(os.path.join(self.images_dir, img_name))
                self.targets.append(os.path.join(self.targets_dir, self.split, file))
        # for file_name in os.listdir(img_dir):
        #     target_name_prefix = file_name.split("_leftImg8bit")[0]
        #     target_name = f'{target_name_prefix}.png'
        #     self.images.append(os.path.join(img_dir, file_name))
        #     self.targets.append(os.path.join(target_dir, target_name))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.rgb == True:
            image = Image.open(self.images[index]).convert('RGB')
        if self.gray == True:
            try:
                image = ImageOps.grayscale(image)  
            except:
                logging.info('Please use the right type of your image (PIL)')
        elif self.hsv:
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = image[...,0:-1]
            image = Image.fromarray(image)
        
        target = Image.open(self.targets[index])
        if self.split == 'train':
            image, target = Crop()._sync_transform(image, target)
        elif self.split == 'val' or self.split == 'test':
            image, target = Crop()._val_sync_transform(image, target)
        image, target = self.transform(image, target)
        
        path = self.images[index]
        
        return image, target, path

    def __len__(self) -> int:
        return len(self.images)




class PascalContextHED33classes(Dataset):
    PascalContextClass = namedtuple('PascalContextClass', ['name',
                                                           'id', 
                                                           'train_id',
                                                           'color'
                                                           ])
    labels = [
        PascalContextClass('void',          0,  255,(  0,  0,  0)),
        PascalContextClass('aeroplane',     2,    0,(128,  0,  0)),
        PascalContextClass('bicycle',      23,    1,(  0,128,  0)),
        PascalContextClass('bird',         25,    2,(128,128,  0)),
        PascalContextClass('boat',         31,    3,(  0,  0,128) ),
        PascalContextClass('bottle',       34,    4,(128,  0,128)),
        PascalContextClass('bus',          45,    5,(  0,128,128)),
        PascalContextClass('car',          59,    6,(128,128,128)),
        PascalContextClass('cat',          65,    7,( 64,  0,  0)),
        PascalContextClass('chair',        72,    8,(192,  0,  0)),
        PascalContextClass('cow',          98,    9,( 64,128,  0)),
        PascalContextClass('diningtabel', 397,   10,(192,128,  0)),
        PascalContextClass('dog',         113,   11,( 64,  0,128)),
        PascalContextClass('horse',       207,   12,(192,  0,128)),
        PascalContextClass('motorbike',   258,   13,( 64,128,128)),
        PascalContextClass('person',      284,   14,(192,128,128)),
        PascalContextClass('pottedplant', 308,   15,(  0, 64,  0)),
        PascalContextClass('sheep',       347,   16,(128, 64,  0)),
        PascalContextClass('sofa',        368,   17,(  0,192,  0)),
        PascalContextClass('train',       416,   18,(128,192,  0)),
        PascalContextClass('tvmonitor',   427,   19,(  0, 64,128)),
        PascalContextClass('sky',         360,   20,(  0,192, 64)),
        PascalContextClass('grass',       187,   21,(  0,  0,192)),
        PascalContextClass('ground',      189,   22,(128,  0,192)),
        PascalContextClass('road',        324,   23,( 64,128,192)),
        PascalContextClass('building',     44,   24,(192, 64,  0)),
        PascalContextClass('tree',        420,   25,( 64,192,  0)),
        PascalContextClass('water',       445,   26,(192, 64, 64)),
        PascalContextClass('mountain',    259,   27,( 64,  0, 64)),
        PascalContextClass('wall',        440,   28,(128, 64,128)),
        PascalContextClass('floor',       158,   29,(128,  0, 64)),
        PascalContextClass('track',       415,   30,(128, 64,192)),
        PascalContextClass('keyboard',    220,   31,(  0,128,192)),
        PascalContextClass('ceiling',      68,   32,(192,192,  0))
    ]
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
  
    # for i in range(len(labels)):
    #     if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
    #         ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        #if labels[i].train_id not in ignore_in_eval_ids:
        train_ids.append(labels[i].train_id)
        color_palette_train_ids[labels[i].train_id] = labels[i].color
        train_id2id.append(labels[i].id)
       
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_2label = {label.color: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    # hed   mean: [0.8532917]
    #       std: [0.2307121] 

    # EED rgb  mean: [0.5605215, 0.5426824, 0.5156801]
    #          std:  [0.2621590, 0.2625541, 0.2722014]

    # EED gray  mean: [0.5470292]
    #           std:  [0.2618797]

    # EED hs mean: [0.3124114, 0.4446574]
    #          std: [0.1950817, 0.2582020]


    def __init__(self,
                    img_root: str,
                    label_root: str,
                    split: str,
                    transform: Dict = None,
                    mode: str = "gtFine",
                    #target_type: str = "labelTrainIDs",
                    train_on_train_id: bool = True,
                    gray: bool=False,
                    rgb:bool=True
                    ) -> None:
        """
        Pascal Context dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' 
        self.train_on_train_id = train_on_train_id
        self.gray = gray
        self.rgb = rgb
        self.hsv = False
        # build transformation
        trafo_list = []
        if self.split == 'train' and 'random_crop' in transform:
            trafo_list.append(RandomCrop(tuple(transform['random_crop'])))
        
        
        trafo_list.append(ToTensor())
        if type(transform['normalization_mean']) == list:
            trafo_list.append(Normalize(tuple(transform['normalization_mean']),
                                        tuple(transform['normalization_var'])))
            if len(transform['normalization_mean']) == 1:
                self.gray = True
            if len(transform['normalization_mean']) == 2:
                self.hsv = True
        else:
            exit("Error: normalizaion parameters need to be of type list")
            # trafo_list.append(Normalize(transform['normalization_mean'],
            #                             transform['normalization_var']))
            # self.gray = True
        self.transform = Compose(trafo_list)

        # data root paths
        if self.split == 'val_train':
            exit("Please use the val split since a seperate test split exists")
            self.split = 'train'
        self.images_dir = os.path.join(self.root)
        self.targets_dir = os.path.join(self.target_root)

        self.images = []
        self.targets = []

        # data paths
        # for sequence in os.listdir(self.images_dir):
        #     if (self.split == 'train' or self.split == 'val') and sequence not in ignore_sequences:
        #         img_dir = os.path.join(self.images_dir, sequence)
        #         target_dir = os.path.join(self.targets_dir, sequence)
        for root, dirs, files in os.walk(os.path.join(self.targets_dir, self.split)):
            for file in files:
                img_name = f'{file.split(".")[0]}.jpg'
                
                self.images.append(os.path.join(self.images_dir, img_name))
                self.targets.append(os.path.join(self.targets_dir, self.split, file))
        # for file_name in os.listdir(img_dir):
        #     target_name_prefix = file_name.split("_leftImg8bit")[0]
        #     target_name = f'{target_name_prefix}.png'
        #     self.images.append(os.path.join(img_dir, file_name))
        #     self.targets.append(os.path.join(target_dir, target_name))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.rgb == True:
            image = Image.open(self.images[index]).convert('RGB')
            fill_img_value=(255,255,255)
        if self.gray == True:
            try:
                image = ImageOps.grayscale(image)  
                fill_img_value=255
            except:
                logging.info('Please use the right type of your image (PIL)')
        elif self.hsv:
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = image[...,0:-1]
            image = Image.fromarray(image)
            fill_img_value=(179,255)
            
        
        target = Image.open(self.targets[index])
        if self.split == 'train':
            image, target = Crop(fill_img_value=fill_img_value)._sync_transform(image, target)
        elif self.split == 'val' or self.split == 'test':
            image, target = Crop(fill_img_value=fill_img_value)._val_sync_transform(image, target)
        image, target = self.transform(image, target)
        
        path = self.images[index]
        
        return image, target, path

    def __len__(self) -> int:
        return len(self.images)
