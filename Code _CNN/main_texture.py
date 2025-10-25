import datetime
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import wandb
import yaml

import utils.iou as iou
from utils import evaluate_utils
from utils.training_utils import get_dataloader, get_model, get_lr_scheduler
from utils.training_utils import load_checkpoint, save_model



def validate_model(model, config, epoch, valid_dl, loss_func, device, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.
    # setup
    confusion_matrix = iou.generate_matrix(config.num_classes)
    count = 0
    with torch.inference_mode():
        # loop through validation set:
        for i, (images, labels, paths) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)['out']
            logits = outputs.data.cpu()
            pred = torch.squeeze(torch.argmax(logits, dim=1))
            # Log one batch of images to the dashboard, always same batch_idx.
            color_palette = valid_dl.dataset.color_palette_train_ids
            table = wandb.Table(columns=["image", "pred", "target"])
            # if i == batch_idx and log_images:
            #     log_image_table(table,
            #                     images,
            #                     pred,
            #                     labels,
            #                     color_palette)

            val_loss += loss_func(outputs, labels)*labels.size(0)

            pred_array = pred.numpy()
            gt_array = torch.squeeze(labels.cpu()).numpy()
            class_names = {}
            for label in range(config.num_classes):
                class_names[label] = (valid_dl.dataset.train_id2label[label].name)

            # calculate iou
            for pred_i, gt_i, path_i in zip(pred_array, gt_array, paths):
                if count == len(valid_dl) - 1 and log_images:
                    log_media(Image.open(path_i), pred_i, gt_i, class_names)
                iou.evaluate_pair(pred_i, gt_i, confusion_matrix, 255)

                if i == batch_idx and log_images:
                    evaluate_utils.dump_prediction_mask(pred_i,
                                                        config.pred_root,
                                                        valid_dl.dataset.color_palette_train_ids,
                                                        epoch,
                                                        iter=os.path.basename(
                                                            evaluate_utils.replace_jpg_with_png(path_i)
                                                            )
                                                        )
            del outputs, images
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


def log_image_table(table, images, predicted, labels, color_palette):
    "Log a wandb.Table with (img, pred, target, scores)"
    # Create a wandb Table to log images, labels and predictions to
    # table = wandb.Table(columns=["image", "pred", "target"])
    for img, pred, targ in zip(images.to("cpu"),
                               predicted.to("cpu"),
                               labels.to("cpu")
                               ):

        colored_pred = Image.fromarray(pred.numpy().astype("uint8")).convert('P')
        colored_pred.putpalette(color_palette)
        colored_target = Image.fromarray(targ.numpy().astype("uint8")).convert('P')
        colored_target.putpalette(color_palette)

        table.add_data(wandb.Image(np.moveaxis(img.numpy()*255, 0, -1)),
                       wandb.Image(colored_pred),
                       wandb.Image(colored_target),
                       )
    wandb.log({"predictions_table": table})

def log_media(img, pred, gt, class_names):
    original_image = img

    wandb.log(
        {"predictions": wandb.Image(original_image, masks={
            "predictions": {
                "mask_data": pred,
                "class_labels": class_names
            },
            "ground_truth": {
                "mask_data": gt,
                "class_labels": class_names
            }
        })})


def train(wb_config, wb_run):
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

    # Get the data
    train_dl = get_dataloader(config.dataset_name,
                              config.dataset_train_root,
                              config.dataset_label_root,
                              split='train',
                              transforms=config.transform_train,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              train_on_train_id=config.generate_train_ids_from_labels
                              )
    valid_dl = get_dataloader(config.dataset_name,
                              config.dataset_test_root,
                              config.dataset_test_label_root,
                              # split='train_testsplit' or 'val
                              split=config.split,
                              transforms=config.transform_test_split,
                              batch_size=config.batch_size_test,
                              num_workers=config.num_workers,
                              train_on_train_id=config.generate_train_ids_from_labels
                              )
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # define class weighting
    if len(config.weights) == 0:
        class_weights = np.ones(config.num_classes)
    else:
        class_weights = np.ones(len(config.weights))/np.array(config.weights)

    # Load model architecture
    model = get_model(config, device)

    # Define the loss, optimizer and lr scheduler
    loss_func = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32).to(device),
        ignore_index=255
    )
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

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    # wandb.watch(model, log_freq=config.log_freq, log="all")

    # Result directory handling
    try:
        os.makedirs(config.pred_root, exist_ok=pred_dir_exists_ok)
        print(f"Results are stored in Directory '{config.pred_root}'.")
        # if config.blend:
        #     save_dir = os.path.join(config.pred_root, f"blended")
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #         print("Created:", save_dir)
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
    for epoch in tqdm(range(start_epoch, config.epochs)):
        model.train()
        for step, (images, labels, paths) in tqdm(enumerate(train_dl)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)['out']
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(images)

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
                   run=wb_run)

        # Validate training
        val_loss, miou = validate_model(model,
                                            config,
                                            epoch,
                                            valid_dl,
                                            loss_func,
                                            device,
                                            log_images=config.log_images)  # formerly was set to True
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
                       best_miou=best_miou)

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
    # name = "texture_gray_CS19_deeplabv3resnet18_512x512_Adam_cs_extra_train_val"
    name = "texture_colored_CS19_deeplabv3bagnet9_512x512_Adam_cs_extra_train_val"
    # name = "texture_HS_CS19_deeplabv3resnet18_512x512_Adam_cs_extra_train_val"
    date = datetime.datetime.now()
    prod_date = f"{date.year}_{date.month:02d}_{date.day:02d}"
    run = wandb.init(
        project="texture_19cl_CS_train_extra_val",
        name=name)

    wandb.config.exp_name = f"{name}_{run.id}"
    wandb.config.pred_root = f"experiments/{wandb.config.exp_name}_{prod_date}/"

    if wandb.config.input_channels != len(wandb.config.transform_test_split["normalization_mean"]):
        exit(f"input channels ({wandb.config.input_channels}) does not \
            match normalization parameter amount ({len(wandb.config.transform_test_split['normalization_mean'])})")

    train(wandb.config, run)
    # 🐝 Close your wandb run
    wandb.finish()
