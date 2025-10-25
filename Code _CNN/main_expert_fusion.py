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

from models.fuse_alex_net_model import FusionAlexNetModel
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



def validate_model(model, config, epoch, valid_dl, loss_func, device, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
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
                # if count == len(valid_dl) - 1 and log_images:
                #     log_media(Image.open(path_i), pred_i, gt_i, class_names)
                iou.evaluate_pair(pred_i, gt_i, confusion_matrix, 255)

                # plot heatmaps
                if i == batch_idx and log_images:
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
                                                        valid_dl.dataset.color_palette_train_ids,
                                                        epoch,
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

        with open(os.path.join(config.pred_root, f"mIoU_only_{config.exp_name}.txt"), "a") as fh:
            fh.write(f"{epoch}, {miou}\n")

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

    return val_loss / len(valid_dl.dataset), miou





def train_attention_net(wb_config, wb_run):
    # Copy your config
    config = wb_config
    pred_dir_exists_ok = False

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
    train_dl = get_dataloader(config.dataset_name,
                              softmax_root_list,
                              config.dataset_label_root,
                              split='train',
                              transforms=config.transform_train,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              train_on_train_id=config.generate_train_ids_from_train_labels,
                              expert_dicts=config.expert_dicts
                              )
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
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # Load experts architecture
    # expert_list = get_experts(config.expert_dicts, device)

    # Load attention net
    inchannels = 0
    for expert in config.expert_dicts:
        inchannels += expert['num_classes']
    if inchannels == 0:
        exit("Experts are not returning enough values at least one channel needed")

    log_probs = (config.loss == "NLLLoss")  # log

    model = get_model(config, device)


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

    if config.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.99)
    elif config.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = get_lr_scheduler(config.lr_policy,
                                    config.num_classes,
                                    optimizer,
                                    train_dl,
                                    config.batch_size)

    # Load model checkpoint
    start_epoch = 0
    if config.ckpt_path != None:
        start_epoch = load_checkpoint(config,
                                      model,
                                      optimizer,
                                      lr_scheduler,
                                      device)
        pred_dir_exists_ok = True

    wandb.watch(model, log_freq=config.log_freq, log="all")

    # Result directory handling
    try:
        os.makedirs(config.pred_root, exist_ok=pred_dir_exists_ok)
        print(f"Results are stored in Directory '{config.pred_root}'.")
        if config.blend:
            save_dir = os.path.join(config.pred_root, f"blended", str(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print("Created:", save_dir)
    except OSError as error:
        print(f"Directory '{config.pred_root}' already exists or can not be created. {error}. Will exit now...")
        exit()

    # Save config as yaml
    with open(os.path.join(config.pred_root, f"config_{config.exp_name}.yaml"), "w") as fh:
        yaml.dump(dict(wandb.config), fh)

    # Training routine
    example_ct = 0
    step_ct = 0
    best_miou = 0
    model.to(device)
    for epoch in tqdm(range(start_epoch, config.epochs), desc="Epochs"):
        model.train()
        train_loss_list = []
        for step, (cat_softmax, labels, paths) in tqdm(enumerate(train_dl), desc="Batch steps"):
            cat_softmax, labels = cat_softmax.to(device), labels.to(device)

            # calculate fusion
            outputs = model(cat_softmax)

            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(cat_softmax)

            try:
                lr_scheduler.step()
            except TypeError:
                print("LR is of wrong type. Most probably it is now complex. Will stop training now...")
                optimizer.param_groups[0]["lr"] = 0
                break
            metrics = {"train/train_loss": train_loss,
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                        "train/lr": lr_scheduler.get_last_lr()[0]
                        }
            train_loss_list.append(train_loss.item())
            if step + 1 < n_steps_per_epoch:
                # 🐝 Log train metrics to wandb
                wandb.log(metrics)

            step_ct += 1

        # save latest model
        save_basename = config.exp_name + '_latest' + ".pth"
        save_path = os.path.join(config.pred_root, save_basename)
        print('Saving checkpoint', save_path)
        save_model(epoch,
                    model,
                    optimizer,
                    train_loss,
                    lr_scheduler,
                    save_path,
                    run=wb_run,
                    train_loss_list=train_loss_list)

        # Validate training
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        val_loss, miou = validate_model(model,
                                        config,
                                        epoch,
                                        valid_dl,
                                        loss_func,
                                        device,
                                        log_images=True)  # log_images=(epoch == (config.epochs-1)

        # Update best miou and save model
        if miou > best_miou:
            best_miou = miou

            save_basename = config.exp_name + '_' + str(epoch) + '_compl' + ".pth"
            save_path = os.path.join(config.pred_root, save_basename)
            save_model(epoch,
                        model,
                        optimizer,
                        train_loss,
                        lr_scheduler,
                        save_path,
                        run=wb_run,
                        best_miou=best_miou,
                        train_loss_list=train_loss_list)

            save_basename = config.exp_name + '_' + 'best_model' + ".pth"
            save_path = os.path.join(config.pred_root, save_basename)
            save_model(epoch,
                        model,
                        optimizer,
                        train_loss,
                        lr_scheduler,
                        save_path,
                        run=wb_run,
                        best_miou=best_miou)

        # 🐝 Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss,
                        "val/val_miou": miou,
                        "val/val_best_miou": best_miou}
        wandb.log({**metrics, **val_metrics})

        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, mIoU: {miou:.2f}")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # 🐝 initialise a wandb run
    name = "fuse_colored_texture_anisodiff_resnet2layer_uniWeights_carlasettings_normalized"
    date = datetime.datetime.now()
    notes = ""
    prod_date = f"{date.year}_{date.month:02d}_{date.day}"
    run = wandb.init(
        project="expert_attention_19cl_CS",
        name=name,
        notes=notes,
        # mode="disabled"
        )

    wandb.config.exp_name = f"{name}"
    wandb.config.pred_root = f"experiments/{wandb.config.exp_name}_{prod_date}_{run.id}/"

    for expert in wandb.config.expert_dicts:
        if len(expert['softmax_transform']['normalization_mean']) != wandb.config.num_classes:
            exit(f"Expert {expert['expert_type']}: \
                normalization mean dimension ({expert['softmax_transform']['normalization_mean']}) \
                does not correlate to number of classes ({wandb.config.num_classes})")


    train_attention_net(wandb.config, run)
    # 🐝 Close your wandb run
    wandb.finish()
