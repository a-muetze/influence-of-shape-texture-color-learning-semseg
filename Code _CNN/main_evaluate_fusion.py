import datetime
import math
import os
import random
import sys
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
import yaml


from utils import iou
from utils import evaluate_utils
from utils.training_utils import get_dataloader, get_loss, get_lr_scheduler, get_model, load_checkpoint, save_model

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self





def get_experts(expert_dict_list, device):
    """Load experts from checkpoints

    Args:
        expert_list (list): List containing model_class, ckpt path and number of input and output channels

        device (string): cuda or cpu. Model will be send to this device after loading.
    """
    model_list = []
    for k, expert_dict in enumerate(expert_dict_list):
        expert_config = AttrDict(expert_dict)
        expert = (get_model(expert_config, device))
        if expert_config.ckpt_path != None:
            checkpoint = torch.load(expert_config.ckpt_path, map_location=device)
            expert.load_state_dict(checkpoint['state_dict'], strict=True)
        expert.to(device)
        expert.eval()
        model_list.append(expert)
        print(f"Expert {k+1} from {len(expert_dict_list)} was loaded")
    return model_list




def validate_model(config, epoch, valid_dl, loss_func, device, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"

    model = get_model(config, device)
    if config.ckpt_path != None:
        checkpoint = torch.load(config.ckpt_path, map_location=device)
        if not checkpoint:
            exit(f"Failed to load checkpoint: {config.ckpt_path}")
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.eval()
    val_loss = 0.
    # setup
    confusion_matrix = iou.generate_matrix(config.num_classes)
    count = 0
    with torch.inference_mode():
        # loop through validation set:
        for i, (cat_softmax, labels, paths) in enumerate(valid_dl):
            cat_softmax, labels = cat_softmax.to(device), labels.to(device)

            # Forward pass
            outputs = model(cat_softmax)#['out']

            cumul_logits = outputs.data.cpu()
            pred = torch.squeeze(torch.argmax(cumul_logits, dim=1))


            val_loss += loss_func(outputs, labels)*labels.size(0)

            pred_array = pred.numpy()
            gt_array = torch.squeeze(labels.cpu()).numpy()
            class_names = {}
            for label in range(config.num_classes):
                class_names[label] = (valid_dl.dataset.train_id2label[label].name)

            # collect heatmap
            expert_weights = torch.tensor_split(model.net.expert_weights, len(config.expert_dicts), dim=1)
            weight_array = [torch.squeeze(weight.cpu()).numpy() for weight in expert_weights]

            # calculate iou and dump heatmaps
            idx = 0
            for pred_i, gt_i in zip(pred_array, gt_array):
                iou.evaluate_pair(pred_i, gt_i, confusion_matrix, 255)

                # plot heatmaps
                if log_images:# and i == batch_idx: 
                    for weight, expert_path in zip(weight_array, paths):
                        evaluate_utils.dump_heatmap_mask(weight[idx],
                                                            config.pred_root,
                                                            epoch,
                                                            iter=expert_path[idx].split('/')[2]
                                                                + "_"
                                                                + os.path.basename(expert_path[idx].removesuffix('.pt')+'.png'),
                                                            dump_type="heatmap"
                                                            )
                    colored_pred = evaluate_utils.dump_prediction_mask(pred_i,
                                                        config.pred_root,
                                                        color_mapping=valid_dl.dataset.color_palette_train_ids,
                                                        epoch=f"{epoch}_{config.testset_name}",
                                                        iter=os.path.basename(paths[0][idx].removesuffix('.pt')+'.png')
                                                        )

                    # if config.blend:
                    #     blended_save_path = os.path.join(config.pred_root, f"blended", str(epoch))
                    #     if not os.path.exists(blended_save_path):
                    #         os.makedirs(blended_save_path)
                    #         print("Created:", blended_save_path)
                    #     img_rgb = Image.open(paths[0][idx]) # this does not make sense as it is a softmax tensor saved as .pt
                    #     blend = Image.blend(img_rgb.convert("RGBA"), colored_pred.convert("RGBA"), alpha=0.5)
                    #     blend.save(os.path.join(blended_save_path, f'{str(iter)}'))
                idx += 1
            del outputs, cat_softmax
            count += 1
            print("\rImages Processed: {}".format(count*config.batch_size_test), end=' ')
            sys.stdout.flush()

        # calculate miou and save results
        classScoreList = {}
        class_names = {}
        for label in range(config.num_classes):
            class_names[label] = (valid_dl.dataset.train_id2label[label].name)
            classScoreList[class_names[label]] = iou.get_iou_score_for_label(label, confusion_matrix)
        print("\n")
        miou = iou.get_score_average(classScoreList)


        with open(os.path.join(config.pred_root, f"mIoU_{config.exp_name}.txt"), "a") as fh:
            fh.write(f"=================================\n")
            fh.write(f"Epoch: {epoch} \n")
            iou.print_class_scores(classScoreList, class_names, fh)
            miou_color = iou.get_color_entry(iou.get_score_average(classScoreList)) + "{avg:5.3f}".format(
                avg=iou.get_score_average(classScoreList)) + iou.Style.ENDC
            fh.write("--------------------------------\n")
            fh.write("Score Average : " + "{avg:5.3f}".format(avg=miou) + " (mIoU)\n")
            fh.write("--------------------------------\n")
            print("--------------------------------")
            print("Score Average : " + miou_color + " (mIoU)")
            print("--------------------------------")
    print(f'Model performance on {config.testset_name} images: {miou}')
    return val_loss / len(valid_dl.dataset), miou





def evaluate_fusion(cfg, epoch, testset_name, log_images=False ):
    # Copy your config
    config = cfg

    # controll the randomness
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    # gather pathes of experts
    softmax_root_list =[expert['softmax_path'] for expert in config.expert_dicts]
    # Get the data

    valid_dl = get_dataloader(config.dataset_name,
                              softmax_root_list,
                              config.dataset_test_label_root,
                              # split='train_testsplit' or 'val
                              split=config.split,
                              transforms=config.transform_test_split,
                              batch_size=config.batch_size_test,
                              num_workers=config.num_workers,
                              train_on_train_id=config.generate_train_ids_from_split_labels,
                              expert_dicts=config.expert_dicts
                              )

    # Load experts architecture
    # expert_list = get_experts(config.expert_dicts, device)

    # Load attention net
    inchannels = 0
    for expert in config.expert_dicts:
        inchannels += expert['num_classes']
    if inchannels == 0:
        exit("Experts are not returning enough values at least one channel needed")

    log_probs = (config.loss == "NLLLoss")  # log

    # define class weighting
    if len(config.weights) == 0:
        class_weights = np.ones(config.num_classes)
    else:
        class_weights = np.ones(len(config.weights))/np.array(config.weights)

    # Define the loss, optimizer and lr scheduler
    loss = get_loss(config.loss)
    loss_func = loss(weight=torch.as_tensor(
        class_weights, dtype=torch.float32).to(device),
        ignore_index=255)
    try:
        os.makedirs(config.pred_root)
        print(f"Results are stored in Directory '{config.pred_root}'.")
    except OSError as error:
        print(f"Directory '{config.pred_root}' already exists or can not be created. {error}. Will exit now...")
        exit()
    # Save config as yaml
    with open(os.path.join(config.pred_root, f"config_{config.exp_name}.yaml"), "w") as fh:
        yaml.dump(dict(config), fh)


    # Validate training
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    val_loss, miou = validate_model(config,
                                    epoch,
                                    valid_dl,
                                    loss_func,
                                    device,
                                    log_images=True)  # log_images=(epoch == (config.epochs-1)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    name = "fuse_colored_texture_anisodiff_resnet2layer_uniWeights_normalized"
    prod_date = "2024_09_26"
    uid = "CgG6V3GubvTgmsDhh6aWCW"
    date = datetime.datetime.now()
    new_prod_date = f"{date.year}_{date.month:02d}_{date.day}"
    notes = ""
    run = wandb.init(
        project="expert_attention_19CS_normalisation_uniform_inits",
        name=name,
        notes=notes,
        mode="disabled"
        )
    eval_dataset_name = "Cityscapes"
    config=AttrDict({
        "split": "val",
        "testset_name" : "CityscapesVal",
        "dataset_name": 'CityscapesFusionSoftmax19classes',
        # "dataset_name": 'Softmax19classes',
        "model_class": 'fusion',
        "deeplab_backbone": 'deeplabv3resnet_2layer',
        "eps": 0.000000000000000000000000000000000000000000001,
        "reset_weights": True,
        "uniform_init": True,
        "exp_name": name,
        "ckpt_path": f"experiments/{name}_{prod_date}_{uid}/{name}_best_model.pth",
        "pred_root": f"experiments/{name}_{uid}_{new_prod_date}",
        "dataset_test_label_root": "/home/datasets/Cityscapes",
        "input_channels": 38,
        "num_classes": 19,
        "num_workers": 8,
        "optim": "Adam",
        "seed": 42,
        "dataset_test_label_root": "/home/datasets/Cityscapes",
        "transform_test_split": {},
        "batch_size_test": 16,
        "generate_train_ids_from_split_labels": False,
        "loss": 'NLLLoss',
        "weights": [],
        "expert_dicts":
            [
            {"expert_type": "texture_colored",
                "model_class": "deeplabv3resnet18",
                "softmax_path": "data/softmax/texture_colored_CS19_deeplabv3resnet18_512x512_cs_extra_train_val_36uw431c_2023_09_06/161/Cityscapes",
                "num_classes": 19,
                "softmax_transform": {
                                    "normalization_mean":[  0.4088935,     0.0103979,     0.2403387,     0.0068888,
                                                            0.0353997,     0.0000924,     0.0003775,     0.0009531,
                                                            0.1217313,     0.0149821,     0.0273380,     0.0007492,
                                                            0.0000685,     0.0357671,     0.0304586,     0.0464466,
                                                            0.0126945,     0.0002257,     0.0062040],
                                    "normalization_var": [  0.4595941,     0.0398623,     0.3437814,     0.0290190,
                                                            0.1013082,     0.0006615,     0.0032343,     0.0090546,
                                                            0.2364252,     0.0554953,     0.1167346,     0.0037226,
                                                            0.0004566,     0.1090197,     0.0895694,     0.1260094,
                                                            0.0437872,     0.0012959,     0.0274537]
                                
                        }
            },
            {"expert_type": "shape_anisodiff",
            "model_class": "deeplabv3resnet18",
            "softmax_path": "data/softmax/contour_anisodiff_CS19_deeplabv3resnet18_512x512_pseudoVal_1gpag6aw_2023_09_15/196/Cityscapes",
            "num_classes": 19,

            "softmax_transform": {
                                ##### softmax normalizeation for EED on Cityscapes
                                "normalization_mean": [ 0.4127309,     0.0460966,     0.1972479,     0.0090963,
                                                        0.0614248,     0.0117656,     0.0011673,     0.0066352,
                                                        0.1275346,     0.0050273,     0.0359360,     0.0099733,
                                                        0.0015958,     0.0608372,     0.0018239,     0.0013836,
                                                        0.0014249,     0.0003258,     0.0079794],
                                "normalization_var": [  0.4765724, 0.1647452, 0.3322418, 0.0377043, 0.1559567, 0.0593846,
                                                        0.0150794, 0.0510134, 0.2655180, 0.0320822, 0.1508082, 0.0538483,
                                                        0.0144647, 0.1856583, 0.0117185, 0.0075018, 0.0053199, 0.0034072,
                                                        0.0466122]


                                }

          }
        ]
    })

    for expert in config.expert_dicts:
        if len(expert['softmax_transform']['normalization_mean']) != expert['num_classes']:
            exit(f"Expert {expert['expert_type']}: \
                normalization mean dimension ({expert['softmax_transform']['normalization_mean']}) \
                does not correlate to number of classes ({expert['num_classes']})")

    if config.split == "train-split":
    # if cfg.split == "train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra":
        eval_dataset_name = eval_dataset_name + "Train_split"
    elif config.split == "val":
        eval_dataset_name = eval_dataset_name + "Val"
    elif config.split == "test":
        eval_dataset_name = eval_dataset_name + "Test"

    
    evaluate_fusion(config, f"best", eval_dataset_name, log_images=True)
    # 🐝 Close your wandb run
    wandb.finish()
