from matplotlib import pyplot as plt
import numpy as np
from datasets.mean_std import calc_dataset_mean_std, calc_datset_class_dist
from datasets.expert_datasets import  CityscapesAnisodiffHS19classes, CityscapesHEDBlackEdges19classes, CityscapesHSV19classes, CityscapesTextureHS
from datasets.expert_datasets import CityscapesTexture2, CityscapesRgb19classes, Softmax19classes
from datasets.expert_datasets import PascalContext33classes, PascalContextHED33classes
## texture in contour
dataset = CityscapesTexture2("/home/xxx//texture-voronoi-diagrams/Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour",
                        "/home/xxx/texture-voronoi-diagrams//Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour_label",
                        split="train",
                        transform={"normalization_mean": [0,0,0],
                                   "normalization_var": [1,1,1]},
                        )
channels = 3

# texture in contour gray
# dataset = CityscapesTexture2("/home/xxx/texture-voronoi-diagrams/Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour",
#                         "/home/xxx/texture-voronoi-diagrams/Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour_label",
#                         split="train",
#                         transform={"normalization_mean": 0,
#                                    "normalization_var": 1},
#                         )
# channels = 1


## oracle
# dataset = CityscapesRgb19classes("/home/datasets/Cityscapes",
#            "/home/datasets/Cityscapes",
#             split="train",
#             target_type="labelIds",
#             train_on_train_id = False,
#             transform={"normalization_mean": [0,0,0],
#                         "normalization_var": [1,1,1]},
#     )
# channels = 3


## HED
# dataset = CityscapesHEDBlackEdges19classes("/home/xxx/data/cityscapes_hed_black_edges",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": 0,
#                                    "normalization_var": 1},
#                         )
# channels = 1

## anisodiff gray
# dataset = CityscapesHEDBlackEdges19classes("/home/xxx/data/anisodiff/cityscapes/8192/leftImg8bit",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": 0,
#                                    "normalization_var": 1},
#                         )
# channels = 1

## anisodiff RGB
# dataset = CityscapesHEDBlackEdges19classes("/home/xxx/data/anisodiff/cityscapes/8192/leftImg8bit",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": [0,0,0],
#                                    "normalization_var": [1,1,1]},
#                         )
# channels = 3



## color HS(V)
# dataset = CityscapesHSV19classes("/home/datasets/Cityscapes",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": [0 for i in range(2)],
#                                    "normalization_var": [1 for i in range(2)]},
#                         )
# channels = 2


## color (HS)V
# dataset = CityscapesHSV19classes("/home/datasets/Cityscapes",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": 0,
#                                    "normalization_var": 1},
#                         )
# channels = 1


## Texture HS(V)
# dataset = CityscapesTextureHS("/home/xxx//texture-voronoi-diagrams/Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour",
#                         "/home/xxx/texture-voronoi-diagrams//Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour_label",
#                         split="train",
#                         transform={"normalization_mean": [0 for i in range(2)],
#                                    "normalization_var": [1 for i in range(2)]},
#                         )
# channels = 2

## Anisodiff HS(V)
# dataset = CityscapesAnisodiffHS19classes("/home/xxx/data/anisodiff/cityscapes/8192/leftImg8bit",
#                          "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": [0 for i in range(2)],
#                                    "normalization_var": [1 for i in range(2)]},
#                         )
# channels = 2

## softmax
# texture_gray_CS19_deeplabv3resnet18_512x512_Adam_cs_extra_train_val_3aaeckjd_2023_09_02/188
# contour_anisodiff_CS19_deeplabv3resnet18_512x512_pseudoVal_1gpag6aw_2023_09_15
# dataset = Softmax19classes("data/softmax/contour_anisodiff_CS19_deeplabv3resnet18_512x512_pseudoVal_1gpag6aw_2023_09_15/196/Cityscapes",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": [0 for i in range(19)],
#                                    "normalization_var": [1 for i in range(19)]},
#                         )
# channels = 19



# dataset = Softmax19classes("data/softmax/contour_hed_blackedge_CS19_deeplabv3resnet18_512x512_pseudoVal_2023_07_31/189/cityscapes_hed_black_edges",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": [0 for i in range(19)],
#                                    "normalization_var": [1 for i in range(19)]},
#                         )
# channels = 19


# dataset = Softmax19classes("data/softmax/contour_hed_blackedge_CS19_deeplabv3resnet18_512x512_pseudoVal_2023_07_31/189/Cityscapes",
#                         "/home/datasets/Cityscapes",
#                         split="train",
#                         transform={"normalization_mean": [0 for i in range(19)],
#                                    "normalization_var": [1 for i in range(19)]},
#                         )
# channels = 19



# dataset = PascalContextHED33classes("/home/xxx/data/pascalcontext_hed_black_edges/VOC2010/JPEGImages",
#                         "/home/.datasets/PASCALVOC/VOCdevkit/VOC2010/SemsegIDImg/",
#                         split="train",
#                         gray=True,
#                         transform={"normalization_mean": 0,
#                                    "normalization_var": 1 },
#                         )
# channels = 1

# dataset = PascalContextHED33classes("/home/.datasets/PASCALVOC/8191",
#                         "/home/.datasets/PASCALVOC/VOCdevkit/VOC2010/SemsegIDImg/",
#                         split="train",
#                         gray=True,
#                         # transform={"normalization_mean": [0,0,0],
#                         #            "normalization_var": [1,1,1] },
#                         transform={"normalization_mean": 0,
#                                    "normalization_var": 1 },
#                         )
# channels = 1



if __name__ == '__main__':

    ## dataset mean and std
    calc_dataset_mean_std(dataset, channels=channels)


    # ## class distribution
    # data = CityscapesRgb19classes("/home/datasets/Cityscapes",
    #        "/home/datasets/Cityscapes",
    #         split="train",
    #         target_type="labelTrainIds",
    #         train_on_train_id = False,
    #         transform={"normalization_mean": [0,0,0],
    #                     "normalization_var": [1,1,1]},
    # )
