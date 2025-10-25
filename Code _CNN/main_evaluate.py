import os
import random
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import invert
from tqdm import tqdm

import utils.iou as iou
from utils import evaluate_utils, model_utils
from utils.training_utils import get_dataloader, get_model
from datasets import expert_datasets



# Evaluation method
# Works for HED, texture and anisodiff models
# TODO: Adapt other methods to get a generic evaluation method
def evaluate_expert(config, testloader, epoch, testset_name, log_images=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config, device)
    if config.ckpt_path != None:
        checkpoint = torch.load(config.ckpt_path, map_location=device)
        if not checkpoint:
            exit(f"Failed to load checkpoint: {config.ckpt_path}")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        # save freshly initialized models
        save_basename = config.exp_name + '_best_model' + ".pth"
        save_path = os.path.join(config.pred_root, save_basename)
        if not os.path.exists(config.pred_root):
            os.makedirs(config.pred_root)
            print("Created:", config.pred_root)
        save_dict = {'epoch': -1,
                    'state_dict': model.state_dict(),
                    }
        torch.save(save_dict, save_path)

    model.eval()
    confusion_matrix = iou.generate_matrix(config.num_classes)

    with torch.inference_mode():
        # loop through validation set:
        for images, labels, paths in tqdm(testloader, unit_scale=config.batch_size_test):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)['out']
            logits = outputs.data.cpu()
            pred = torch.squeeze(torch.argmax(logits, dim=1))
            # Log one batch of images to the dashboard, always same batch_idx.
            #color_palette = testloader.dataset.color_palette_train_ids

            pred_array = pred.numpy()
            gt_array = torch.squeeze(labels.cpu()).numpy()
            class_names = {}
            for label in range(config.num_classes):
                class_names[label] = (testloader.dataset.train_id2label[label].name)

            # calculate iou
            for pred_i, gt_i, path_i in zip(pred_array, gt_array, paths):
                iou.evaluate_pair(pred_i, gt_i, confusion_matrix, 255)
                if log_images:
                    evaluate_utils.dump_prediction_mask(pred_i,
                                                        config.pred_root,
                                                        color_mapping=testloader.dataset.color_palette_train_ids,
                                                        epoch=f"{epoch}_{testset_name}",
                                                        iter=os.path.basename(
                                                            evaluate_utils.replace_jpg_with_png(path_i)
                                                            )
                                                        )
            del outputs, images
            # print("\rImages Processed: {}".format(count*config.batch_size_test), end=' ')
            # sys.stdout.flush()

        # calculate miou and save results
        classScoreList = {}
        class_names = {}
        for label in range(config.num_classes):
            class_names[label] = (testloader.dataset.train_id2label[label].name)
            classScoreList[class_names[label]] = iou.get_iou_score_for_label(label, confusion_matrix)
        print("\n")
        miou = iou.get_score_average(classScoreList)

        with open(os.path.join(config.pred_root, f"mIoU_{config.model}_expert_on_{testset_name}.txt"), "a") as fh:
            fh.write(f"=================================\n")
            fh.write(f"Epoch: {epoch} \n")
            iou.print_class_scores(classScoreList, class_names, fh)
            miou_color = iou.get_color_entry(iou.get_score_average(classScoreList)) + "{avg:5.6f}".format(
                avg=iou.get_score_average(classScoreList)) + iou.Style.ENDC
            fh.write("--------------------------------\n")
            fh.write("Score Average : " + "{avg:5.6f}".format(avg=miou) + " (mIoU)\n")
            fh.write("--------------------------------\n")
            print("--------------------------------")
            print("Score Average : " + miou_color + " (mIoU)")
            print("--------------------------------")

    iou.plot_confusion_matrix(confusion_matrix, class_names, f"{epoch}_{testset_name}", config.pred_root )

    print(f'Model performance on {testset_name} images: {miou}')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



if __name__ == '__main__':
    # name = "contour_anisodiff_CS19_deeplabv3resnet18_512x512_pseudoVal_1gpag6aw"
    # name = "contour_hed_blackedge_CS19_deeplabv3resnet18_512x512_pseudoVal"
    # name = "contour_anisodiff_gray_CS19_deeplabv3resnet18_512x512_pseudoVal_2k5tbo4o"
    # name = "Color_CS19_conv1x1_stack256_256_freshlyInitialized"
    # name = "texture_colored_CS19_deeplabv3resnet18_512x512_cs_extra_train_val_36uw431c"
    # name = "texture_gray_CS19_deeplabv3resnet18_512x512_Adam_cs_extra_train_val_3aaeckjd"
    # name = "texture_colored_CS19_deeplabv3resnet2layer_512x512_Adam_cs_extra_train_val_20use5ib"
    # name = "texture_colored_CS19_deeplabv3resnet2layer_512x512_Adam_cs_extra_train_val_293cpemf"
    # name = "oracle_CS19_deeplabv3resnet18_2anqy7iw"
    # name = "Color_HSmanuell_CS19_LR-4_conv1x1_stack256_256_Custom_FCNHead_3fn6qt1r"
    # name = "Color_CS19_conv1x1_stack256_256_Custom_FCNHead_CSnorm_3p437joa"
    # name = "texture_colored_CS19_deeplabv3bagnet9_512x512_Adam_cs_extra_train_val_lr7f89ru"
    # name = "texture_colored_CS19_deeplabv3bagnet17_512x512_Adam_cs_extra_train_val_2q2ukucj"
    # name = "texture_colored_CS19_deeplabv3bagnet33_512x512_Adam_cs_extra_train_val_vjmkrpz6"
    # name = "Color_V_CS19_conv1x1_stack256_2
    # name = "oracle_gray_CS19_deeplabv3resnet18_243ht3c2"
    # name = "contour_anisodiff_CS19_deeplabv3resnet101_512x512_pseudoVal_1twict2f"
    # name = "contour_anisodiff_HS_CS19_deeplabv3resnet18_512x512_pseudoVal_2c8y9ajc"
    # name = "texture_HS_CS19_deeplabv3resnet18_512x512_Adam_cs_extra_train_val_8bwcg0ps"
    # name = "oracle_HS_CS19_deeplabv3resnet18_3dt8mmei"
    # name = "texture_gray_normadapted_CS19_deeplabv3bagnet17_512x512_Adam_cs_extra_train_val_26pyw32l"
    # name = "freshlyInitialized_oracle_seed1337_CS19_deeplabv3resnet18_cs_extra_train_val"

    name = "contour_anisodiff_seed73_CS19_deeplabv3resnet18_512x512_pseudoVal_2zjp6s50"
    prod_date = "2023_11_22"

    # prod_date = "2023_11_27"
    # prod_date = "2023_07_31"
    # prod_date = "2023_10_12"
    # prod_date = "2023_11_10"
    # prod_date = "2023_09_06"
    # prod_date = "2023_09_02"
    # prod_date = "2023_10_18"
    # prod_date = "2023_10_23"
    # prod_date = "2023_10_25"
    # prod_date = "2023_10_28"
    # prod_date = "2023_11_2"
    # prod_date = "2023_11_09"
    # prod_date = "2023_11_13"
    # prod_date = "2023_11_15"
    # prod_date = "2023_11_10"
    # prod_date = "2023_11_16"
    # prod_date = "2023_11_19"
    # prod_date = "2023_11_15"
    # prod_date = "2023_11_23"
    # prod_date = "2023_12_14"


    # eval_dataset = "deldel"
    # eval_dataset = "AnisodiffOnCityscapes_train_extra_split"
    # eval_dataset = "AnisodiffHSOnCityscapes"
    # eval_dataset = "AnisodiffGrayOnCityscapes"
    eval_dataset = "AnisodiffOnCityscapes"
    # eval_dataset = "AnisodiffOnAnisodiffCS"
    # eval_dataset = "hedOnGrayscaleCityscapes"
    # eval_dataset = "hedOnHedCS"
    # eval_dataset = "delme"
    # eval_dataset = "FreshlyInitializedOnCityscapes"
    # eval_dataset = "coloredTextureOnCityscapes"
    # eval_dataset = "TextureHSOnCityscapes"
    # eval_dataset = "grayTextureOnCityscapes"
    # eval_dataset = "TextureOnCityscapes"
    # eval_dataset = "originalCityscapesVal"
    # eval_dataset = "colorHSmanualOnCityscapes"
    # eval_dataset = "colorOnCityscapes"
    # eval_dataset = "colorVmanualOnCityscapes"
    # eval_dataset = "bagnet9OnCityscapesVal"
    # eval_dataset = "bagnet17OnCityscapesVal"
    # eval_dataset = "bagnet33OnCityscapesVal"
    # eval_dataset = "oracleOnCityscapes"
    # eval_dataset = "oracleGrayOnCityscapes"
    # eval_dataset = "oracleHSOnCityscapes"

    # evaluation of * expert on original Cityscapes dataset:
    config=AttrDict({
            # "model": "contour",
            # "model": "texture",
            # "model": "color",
            "model": "anisodiff",
            # "model": "oracle",
            "model_class": "deeplabv3resnet18",
            # "model_class": "conv1x1",
            # "model_class": "conv1x1Times3",
            # "model_class": "deeplabv3resnet_2layer",
            # "model_class": "deeplabv3bagnet17",
            "input_channels": 3,
            # "input_channels": 1,
            # "input_channels": 2,
            "reset_weights": True,
            "num_classes": 19,
            # "generate_train_ids_from_train_labels": False,
            "generate_train_ids_from_split_labels": False,
            # "generate_train_ids_from_split_labels": True,
            "dataset_name": 'CityscapesRgb19classes',
            # "dataset_name": 'CityscapesTexture2',
            # "dataset_name": 'CityscapesHEDBlackEdges19classes',
            # "dataset_name": 'CityscapesHSV19classes',
            # "dataset_name": 'CityscapesAnisodiffHS19classes',
            "split": "val",
            # "split": "train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra",

            "dataset_test_root": "/home/datasets/Cityscapes",
            "dataset_test_label_root": "/home/datasets/Cityscapes",
            # "dataset_test_root": "/home/xxx/data",
            # "dataset_test_label_root": "/home/xxx/data",
            # "dataset_test_root": "/home/xxx/data/cityscapes_hed_black_edges",
            # "dataset_test_label_root": "/home/datasets/Cityscapes",
            # "dataset_test_label_root": "/home/xxx/data/train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra/pseudo_gt_fine",

            # texture in contour
            # "dataset_test_root": "/home/xxx/texture-voronoi-diagrams/Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour",
            # "dataset_test_label_root": "/home/xxx/texture-voronoi-diagrams/Voronoi_2023_07_31_on_2023_03_03_upsampled_texture_images_in_contour_label",

            # texture wo contour
            # "dataset_test_root": "/home/xxx/texture-voronoi-diagrams/Voronoi_2023_03_03_upsampled_texture_images",
            # "dataset_test_label_root": "/home/xxx/texture-voronoi-diagrams/Voronoi_2023_03_03_upsampled_texture_images_label",

            # anisodiff
            # "dataset_test_root": "/home/xxx/data/anisodiff/cityscapes/8192/leftImg8bit",
            # "dataset_test_label_root": "/home/datasets/Cityscapes",

            # "dataset_test_label_root": "/home/xxx/data/train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra/pseudo_gt_fine",


            "transform_test_split": {"to_tensor": True, #irrelevant. This is ignored by the code and toTensor is always done for images
                                    # # ###### texture wo contour fill
                                    # "normalization_mean": (0.2751, 0.3097, 0.2739),
                                    # "normalization_var": (0.2093, 0.2146, 0.2105),
                                    # # # ###### texture gray wo contour fill (c)
                                    # "normalization_mean": 0.2953,
                                    #  "normalization_var": 0.2114
                                    #  ###### ImageNet Normalization
                                    #  "normalization_mean": [0.485, 0.456, 0.406],
                                    #  "normalization_var": [0.229, 0.224, 0.225]
                                    # # ###### Cityscapes(c)
                                    #  "normalization_mean": [0.2868955, 0.3251328, 0.2838913],
                                    #  "normalization_var": [0.1761364, 0.1809918, 0.1777224]
                                    # ###### texture colored in contour (c)
                                    #  "normalization_mean": [0.2753, 0.3099, 0.274],
                                    #  "normalization_var": [0.2081, 0.2134, 0.2091]
                                    # # # ###### texture gray IN contour (c)
                                    # "normalization_mean": 0.2955,
                                    #  "normalization_var": 0.2102
                                    # # ###### texture HS (c)
                                    #  "normalization_mean": [0.3411435, 0.2125287],
                                    #  "normalization_var": [0.1262746, 0.1077188]
                                    # ####### anisodiff (c)
                                    "normalization_mean": [0.2544, 0.2949, 0.2515],
                                    "normalization_var": [0.1803, 0.1854, 0.1826]
                                    # ###### anisodiff gray (c)
                                    #  "normalization_mean":  0.2778,
                                    #  "normalization_var": 0.1832
                                    # ####### anisodiff HS (c)
                                    # "normalization_mean": [0.328504, 0.2477634],
                                    # "normalization_var": [0.0729014, 0.1185639]
                                    # ###### hed(c)
                                    #  "normalization_mean": 0.9037,
                                    #  "normalization_var": 0.1800
                                    # ###### colorHSmanual (c)
                                    #  "normalization_mean": [0.3287114, 0.1922254],
                                    #  "normalization_var": [0.0873381, 0.0792537]
                                    ##### colorVmanual (c)
                                    #  "normalization_mean": 0.3269421,
                                    #  "normalization_var": 0.1816357
                                     ###### oracle gray (c)
                                    #  "normalization_mean":  0.3090155,
                                    #  "normalization_var": 0.1786242
                                    # ###### oracle HS (c)
                                    #  "normalization_mean": [0.3287114, 0.1922254],
                                    #  "normalization_var": [0.0873381, 0.0792537]
                                    },
            "batch_size": 20,
            "batch_size_test": 2,
            # "batch_size_test": 4,
            # "num_workers": 10,
            "num_workers": 4,
            "weights": [],
            "lr_policy": "linear",
            # "ckpt_path": None,
            "ckpt_path": f"experiments/{name}_{prod_date}/{name}_best_model.pth",
            "pred_root": f"experiments/{name}_{prod_date}",
            "exp_name": name,
            "dump_pred": False,
            "seed": 42,
            "log_freq": 2975,
        })

    # config = wandb.config

    # controll the randomness
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    testloader = get_dataloader(config.dataset_name,
                   config.dataset_test_root,
                   config.dataset_test_label_root,
                   config.transform_test_split,
                   split=config.split,
                   batch_size=config.batch_size_test,
                   num_workers=config.num_workers,
                   train_on_train_id=config.generate_train_ids_from_split_labels,
                   drop_last=(config.split == "train")
                   )


    if config.split == "train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra":
        eval_dataset = eval_dataset + "Train_Extra"
    if config.split == "val":
        eval_dataset = eval_dataset + "Val"

    evaluate_expert(config, testloader, f"best", eval_dataset, log_images=True)

