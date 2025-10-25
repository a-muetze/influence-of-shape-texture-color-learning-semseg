import datetime
import os
import random
from collections import namedtuple
import time
import pickle

import numpy as np
import torch
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import cv2

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

# Trees
import xgboost as xgb

class CityscapesForColorSampling():

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
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)


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
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.train_on_train_id = train_on_train_id
        self.target_type = target_type

        # data root
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split.split('_')[0])
        self.targets_dir = os.path.join(self.root, self.mode, self.split.split('_')[0])

        self.images = []
        self.targets = []

        for city in os.listdir(self.images_dir):
            if (self.split == 'train' or self.split == 'val') and city not in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))
            if self.split == 'train_testsplit' and city in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))

    # def sample_pixel_color(self, num_images, num_samples_per_img):
        # resampled = 0
        # sampled_pixel_list =[[] for i in range(self.num_train_ids if (self.train_on_train_id or self.target_type == "labelTrainIds") else self.num_label_ids )]
        # for index in tqdm(range(num_images)):
        #     img = Image.open(self.images[index]).convert('HSV')
        #     img_np = np.array(img)
        #     target = Image.open(self.targets[index])
        #     np_label = np.array(target) #np.uint8(
        #     if self.train_on_train_id:
        #         # transform segmentation mask according to train_ids
        #         mask = 255*np.ones((np_label.shape), dtype=np.uint8)
        #         for i in np.unique(np_label):
        #             mask[np_label == i] = self.labels[i].train_id
        #         target = mask # Image.fromarray(mask)
        #     else:
        #         target = np_label
        #     x_list = random.sample(range(0, img_np.shape[0]), num_samples_per_img)
        #     y_list = random.sample(range(0, img_np.shape[1]), num_samples_per_img)

        #     for j , (x,y) in enumerate(zip(x_list, y_list)):
        #         while target[x, y] == 255:
        #             x = random.randint(0, img_np.shape[0] - 1)
        #             y = random.randint(0, img_np.shape[1] - 1)
        #             resampled += 1
        #             continue
        #         if img_np[x, y].any() < 0:
        #             print(f"({x},{y}) with value {img_np[x, y]}")
        #         # print(img_np[x, y])
        #         sampled_pixel_list[target[x, y]].append(img_np[x, y])
        # return sampled_pixel_list, resampled

    def sample_pixel_color_per_class(self, class_id, total_samples, num_samples_per_img):
        num_total_samples = 0
        sampled_pixel_list =[]
        while num_total_samples < total_samples:
            index = random.randint(0, len(self.images)-1)
            image = cv2.imread(self.images[index])
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # img = Image.open(self.images[index]).convert('HSV')
            # img_np = np.uint8(np.array(img))
            # img_np = np.array(hsv)
            np_label = cv2.imread(self.targets[index], cv2.IMREAD_GRAYSCALE)
            # np_label = np.uint8(np.array(target))

            # Maskierung des Bildes auf eine Klasse
            if self.train_on_train_id:
                class_pixel_idx = np.where(np_label == train_id2label[class_id])
            else:
                class_pixel_idx = np.where(np_label == class_id)

            class_pixel_coords = [(x,y) for x,y in zip(class_pixel_idx[0], class_pixel_idx[1])]

            if class_pixel_coords:
                coord_list = random.sample(class_pixel_coords, min(num_samples_per_img, len(class_pixel_coords)))

                for j , (x,y) in enumerate(coord_list):
                    sampled_pixel_list.append(hsv[x, y])
                num_total_samples += len(coord_list)
        return sampled_pixel_list


    def sample_all_pixel_color(self, sampled_pixel_list, img_range):
        """Iterate over all images and collect h, s information of the pixel grouped by class id

        Args:
            sampled_pixel_list (list): list of list per class to store h,s values to

        Returns:
            list: list of list per class with h,s values per pixel
        """
        for index in range(img_range[0], img_range[1]):
            img = cv2.imread(self.images[index])
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            np_label = cv2.imread(self.targets[index], cv2.IMREAD_GRAYSCALE)

            # sort pixels values according to class
            pixel_coords = [(y, x) for x in range(img.shape[1]) for y in range(img.shape[0])]
            for pix_coord in tqdm(pixel_coords):
                cl = np_label[pix_coord]
                if cl not in config["ignore_ids"]:
                    sampled_pixel_list[np_label[pix_coord]].append(hsv[pix_coord])
        return sampled_pixel_list



def sampling_pixels_from_cityscapes(config, save_path):

    # controll the randomness if existend
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    # Get the data
    if config["dataset_name"] != 'CityscapesForColorSampling':
        exit("Only CityscapesForColorSampling is implemented yet. Will exit now...")
    City = CityscapesForColorSampling(config["dataset_train_root"],
                                        config["dataset_label_root"],
                                        split='train',
                                        mode='fine',
                                        target_type = "labelTrainIds",
                                        ignore_cities = [],
                                        train_on_train_id=config["generate_train_ids_from_labels"]
                                        )

    hsv_ignore255_list = []
    ignored_summed = 0
    for index in tqdm(range(config["img_range"][0], config["img_range"][1])):
        img = cv2.imread(City.images[index])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        np_label = cv2.imread(City.targets[index], cv2.IMREAD_GRAYSCALE)

        hsv_vec = hsv.reshape( hsv.shape[0]*hsv.shape[1], 3)
        label_vec = np_label.reshape(hsv.shape[0]*hsv.shape[1], 1)
        hsv_cl = np.concatenate((hsv_vec, label_vec), axis=1)
        print(f"pixel per image: {hsv_cl.shape[0]}")
        ignored_summed += np.sum(np.squeeze(label_vec[:]) < 255)
        if np.mean(np.squeeze(label_vec[:]) >= 255) > 0.15:
            print(f"many pixel ignored: { np.sum(np.squeeze(label_vec[:]) >= 255)} ({np.mean(np.squeeze(label_vec[:]) >= 255)*100}%)")

        hsv_ignore255_list.append(hsv_cl[(np.squeeze(label_vec[:]) < 255), : ])

    hsv_ignore255_to_save = np.vstack(hsv_ignore255_list)
    print(f"Sum of ignored pixels: {ignored_summed}")
    start = time.time()
    with open(save_path, 'wb') as fileObj:
        pickle.dump(hsv_ignore255_to_save, fileObj)
    print(f"Took: {time.time() - start} seconds")


    # sampled_pixel_list =[[] for i in range(City.num_train_ids if (City.train_on_train_id or City.target_type == "labelTrainIds") else City.num_label_ids )]

    # # if subset shall be sampled:
    # for c in tqdm(range(len(sampled_pixel_list)), "Class Ids"):
    # for c in tqdm(config.ids, "Class Ids"): #,8,10,13
    #     sampled_pixel_list[c] = City.sample_pixel_color_per_class(c, config.total_samples, config.num_samples_per_img)

    # select if all pixels are used for training
    # sampled_pixel_list = City.sample_all_pixel_color(sampled_pixel_list, )

    # fileObj = open(save_path, 'wb')
    # pickle.dump(sampled_pixel_list, fileObj)
    # fileObj.close()


    # for c, l in enumerate(sampled_pixel_list):
    #     print(f"{len(l)} pixels for class {c}")
    # return sampled_pixel_list



def viualize_pixel_class_dist(df_final):
    plt.style.use('presentation.mplstyle')
    fig = plt.figure(figsize=(10.5,8))
    ax = fig.add_subplot(111) # , projection = '3d')
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    # ax.set_zlabel("v")
    scatter = ax.scatter(df_final[:,0],
               df_final[:,1],
            #    df_final['v'],
               c = df_final[:,3],
               cmap=plt.get_cmap("tab20"),
               alpha=0.5,
               s=20
              )
    # legend
    handles, _ = scatter.legend_elements(prop="colors", alpha=1, num=None)
    unique_names = np.unique(df_final[:,3])
    id_names = [CityscapesForColorSampling.train_id2label[int(class_id)].name for class_id in unique_names ]
    ax.legend(handles, id_names, bbox_to_anchor=(1, 1), loc=2, ncol=2)
    # plt.title(" vs ".join(id_names))

    plt.savefig(f"{'_vs_'.join(id_names)}.png", bbox_inches='tight', transparent=True)


def plot_confusion_matrix(cm, classes, clf, normalized=False, cmap='bone'):
    plt.figure(figsize=[13, 12])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.savefig(f"{clf.__class__.__name__}_confusion_matrix.png", bbox_inches='tight')



######## visualize color distribution in matplotlib
## https://dev.to/codesphere/visualizing-the-color-spaces-of-images-with-python-and-matplotlib-1nmk

def visualizeColorDistribution(path:str, save_path:str)-> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from PIL import Image


    im = Image.open(path)
    px = im.load()
    ax = plt.axes(projection = '3d')

    x = []
    y = []
    z = []
    c = []

    for row in range(0,im.height, 2):
        for col in range(0, im.width, 2):
            pix = px[col,row]
            newCol = (pix[0] / 255, pix[1] / 255, pix[2] / 255)

            if(not newCol in c):
                x.append(pix[0])
                y.append(pix[1])
                z.append(pix[2])
                c.append(newCol)

    ax.scatter(x,y,z, c = c)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, os.path.basename(path)), transparent=True)



import torchvision.transforms as tvtransf

def visualizeColorDistribution2D(root, filen:str, save_path:str)-> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from PIL import Image

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
    def rgb2v_torch(rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        hsv_v = cmax
        return hsv_v

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 20,
        # "font.weight": "bold",  # bold fonts
        "xtick.labelsize": 13,   # large tick labels
        "ytick.labelsize": 13,   # large tick labels
        # "xtick.major.size": 15,
        # "axis.labelpad" : 5,
        "savefig.dpi": 300,     # higher resolution output.
        # "fontsize": 15
    })


    from numpy import moveaxis
    # im = Image.open(os.path.join(root,filen)).convert('RGB')
    # imV = Image.open(os.path.join(root,filen)).convert('L')
    # imV.save(os.path.join("delme_tmp/results_gray", os.path.basename(filen)))
    imcvbgr = cv2.imread(os.path.join(root,filen))
    imcv = cv2.cvtColor(imcvbgr, cv2.COLOR_BGR2RGB)
    hsv_nemo = cv2.cvtColor(imcv, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(imcv, cv2.COLOR_RGB2GRAY)
    h, s, v = cv2.split(hsv_nemo)
    v = np.ones_like(v)*127
    hsv_colors = moveaxis(np.array([h,s,v]),0,2)
    hsv_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB)

    fig = plt.figure()

    from matplotlib import colors
    pixel_colors = imcv.reshape((np.shape(imcv)[0]*np.shape(imcv)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    pixel_colors_hsv = hsv_colors.reshape((np.shape(hsv_colors)[0]*np.shape(hsv_colors)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors_hsv)
    # pixel_colors_hsv = norm(pixel_colors_hsv).tolist()
    # axis = fig.add_subplot(1, 1, 1)
    # axis.scatter(h.flatten(), s.flatten(), facecolors=pixel_colors_hsv, marker=".")
    # axis.set_xlabel("Hue")
    # axis.set_ylabel("Saturation")
    # # axis.set_zlabel("Value")
    # # plt.show()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, os.path.basename(filen)), transparent=True)
    # plt.close()

    # fig = plt.figure()
    # axis = fig.add_subplot(1, 1, 1, projection="3d")

    # r, g, b = cv2.split(imcv)
    # axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    # axis.set_xlabel("\nRed", linespacing=0.6)
    # axis.set_ylabel("\nGreen", linespacing=0.2)
    # axis.set_zlabel("\nBlue", linespacing=0.2)
    # plt.tight_layout()
    # plt.savefig(os.path.join("delme_tmp/results_RGB", os.path.basename(filen)), transparent=True)
    # plt.close()


    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1) #, projection='3d'
    pixel_colors_gray = gray.reshape((np.shape(gray)[0]*np.shape(gray)[1]))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors_gray)
    pixel_colors_gray = norm(pixel_colors_gray).tolist()
    plt.scatter(np.linspace(0, gray.shape[0]*gray.shape[1], gray.shape[0]*gray.shape[1], endpoint=False),
             gray.flatten(),  marker=".",c=pixel_colors_gray, cmap='gray')
    axis.set_xlabel("pixel")
    axis.set_ylabel("gray value")
    plt.tight_layout()
    plt.savefig(os.path.join("delme_tmp/results_gray", os.path.basename(filen)), transparent=True)
    plt.close()


if __name__ == '__main__':

    # visualizeColorDistribution("delme_tmp/RGB/, hamburg_000000_029676_leftImg8bit.png", "delme_tmp/results_RGB")
    visualizeColorDistribution2D("delme_tmp/RGB/", "hamburg_000000_029676_leftImg8bit.png", "delme_tmp/results_HS")
