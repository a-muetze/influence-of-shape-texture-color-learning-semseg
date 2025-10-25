import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import invert
from tqdm import tqdm

import SemsegCityscapes_voronoi as SCv
from helper_scripts.custom_transforms import Compose3, Normalize3, ToTensor3, Normalize, Compose, ToTensor
import utils.iou as iou
from utils import evaluate_utils, model_utils
from utils.training_utils import get_dataloader, get_model
import wandb
from datasets import expert_datasets


def evaluate(expert, model, testloader, epoch, classes):
    pred_rgb_list = []
    pred_hed_list = []
    labels_list = []
    vis_counter = 0

    net = model_utils.features_call(expert, f'checkpoints/fcn_{expert}_{model}.pth')
    net = net.netloader(net_model=model.split('_')[0])

    for j, data in tqdm(enumerate(testloader)):
        inputs, labels, heds = data[0].cuda(), data[1], data[2].cuda()
        labels_list.append(labels)
        gt_root = f'experiments/polygon_images/predicted/ground_truth/{expert}/fcnresnet_beide/'

        logits_rgb = net(inputs)["out"]
        predicted_rgb = torch.argmax(logits_rgb, 1).cpu()
        pred_rgb_list.append(predicted_rgb)
        pred_rgb_root = f'experiments/polygon_images/predicted/{expert}/fcnresnet_beide/hed/'
        del logits_rgb

        logits_hed = net(heds)["out"]
        #logits_hed = net(torch.cat((heds, heds, heds), 1))["out"]
        predicted_hed = torch.argmax(logits_hed, 1).cpu()
        pred_hed_list.append(predicted_hed)
        pred_hed_root = f'experiments/polygon_images/predicted/{expert}/fcnresnet_beide/hed/'
        del inputs, logits_hed

        # if vis_counter % 5 == 0:
        # for pred_i, gt_i in zip(predicted_rgb, labels):
        #     evaluate_utils.dump_prediction_mask(pred_i.cpu().numpy(), pred_rgb_root, epoch=epoch)
        #     #save colorized gt
        #     evaluate_utils.dump_prediction_mask(gt_i.cpu().numpy(), gt_root, epoch=epoch)
        # for pred_i, gt_i in zip(predicted_hed, labels):
        #     evaluate_utils.dump_prediction_mask(pred_i.cpu().numpy(), pred_hed_root, epoch=epoch, iter=j)
        #     evaluate_utils.dump_prediction_mask(gt_i.cpu().numpy(), gt_root, epoch=epoch, iter=j)
        # #    vis_counter += 1

    with open(os.path.join(gt_root, f"mIoU_{expert}.txt"), "a") as fh:
        fh.write(f"=================================\n")
        fh.write(f"Epoch: {epoch} \n")
        miou_rgb = iou.evaluate_img_lists(pred_rgb_list,
                                      labels_list,
                                      7,
                                      255,
                                      class_names=classes,
                                      epoch=epoch,
                                      fh=fh)
    with open(os.path.join(pred_hed_root, f"mIoU_{expert}.txt"), "a") as fh:
        miou_hed = iou.evaluate_img_lists(pred_hed_list,
                                      labels_list,
                                      7,
                                      255,
                                      class_names=classes,
                                      epoch=epoch,
                                      fh=fh)

    print(f'RGB Performance: {miou_rgb} vs HED Performance {miou_hed}')


from main import get_model

def evaluate_contour(config, expert, testloader, epoch, classes):
    pred_rgb_list = []
    pred_hed_list = []
    labels_list = []
    vis_counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = get_model(config, device)

    for j, data in tqdm(enumerate(testloader)):
        inputs, labels, heds = data[0].cuda(), data[1], data[2].cuda()
        labels_list.append(labels)
        gt_root = f'experiments/polygon_images/predicted/ground_truth/{expert}/fcnresnet_beide/'

        logits_rgb = net(inputs)["out"]
        predicted_rgb = torch.argmax(logits_rgb, 1).cpu()
        pred_rgb_list.append(predicted_rgb)
        pred_rgb_root = f'experiments/polygon_images/predicted/{expert}/fcnresnet_beide/hed/'
        del logits_rgb

        logits_hed = net(heds)["out"]
        #logits_hed = net(torch.cat((heds, heds, heds), 1))["out"]
        predicted_hed = torch.argmax(logits_hed, 1).cpu()
        pred_hed_list.append(predicted_hed)
        pred_hed_root = f'experiments/polygon_images/predicted/{expert}/fcnresnet_beide/hed/'
        del inputs, logits_hed

        # if vis_counter % 5 == 0:
        # for pred_i, gt_i in zip(predicted_rgb, labels):
        #     evaluate_utils.dump_prediction_mask(pred_i.cpu().numpy(), pred_rgb_root, epoch=epoch)
        #     #save colorized gt
        #     evaluate_utils.dump_prediction_mask(gt_i.cpu().numpy(), gt_root, epoch=epoch)
        # for pred_i, gt_i in zip(predicted_hed, labels):
        #     evaluate_utils.dump_prediction_mask(pred_i.cpu().numpy(), pred_hed_root, epoch=epoch, iter=j)
        #     evaluate_utils.dump_prediction_mask(gt_i.cpu().numpy(), gt_root, epoch=epoch, iter=j)
        # #    vis_counter += 1

    with open(os.path.join(gt_root, f"mIoU_{expert}.txt"), "a") as fh:
        fh.write(f"=================================\n")
        fh.write(f"Epoch: {epoch} \n")
        miou_rgb = iou.evaluate_img_lists(pred_rgb_list,
                                      labels_list,
                                      7,
                                      255,
                                      class_names=classes,
                                      epoch=epoch,
                                      fh=fh)
    with open(os.path.join(pred_hed_root, f"mIoU_{expert}.txt"), "a") as fh:
        miou_hed = iou.evaluate_img_lists(pred_hed_list,
                                      labels_list,
                                      7,
                                      255,
                                      class_names=classes,
                                      epoch=epoch,
                                      fh=fh)

    print(f'RGB Performance: {miou_rgb} vs HED Performance {miou_hed}')


# Newest evaluation method
# Works for HED, texture and anisodiff models
# TODO: Adapt other methods to get a generic evaluation method
def evaluate_expert(config, testloader, epoch, testset_name, log_images=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config, device)
    if config.ckpt_path != None:
        checkpoint = torch.load(config.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
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
                                                            evaluate_utils.replace_jpg_with_png(path_i))
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
            miou_color = iou.get_color_entry(iou.get_score_average(classScoreList)) + "{avg:5.3f}".format(
                avg=iou.get_score_average(classScoreList)) + iou.Style.ENDC
            fh.write("--------------------------------\n")
            fh.write("Score Average : " + "{avg:5.3f}".format(avg=miou) + " (mIoU)\n")
            fh.write("--------------------------------\n")
            print("--------------------------------")
            print("Score Average : " + miou_color + " (mIoU)")
            print("--------------------------------")

    iou.plot_confusion_matrix(confusion_matrix, class_names, f"{epoch}_{testset_name}")

    print(f'Texture Model performance on {testset_name} images: {miou}')


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



if __name__ == '__main__':
    # name = "texture_CS19_deeplabv3resnet50_textVal"
    name = "contour_anisodiff_CS19_deeplabv3resnet18_512x512_pseudoVal_1gpag6aw"
    prod_date = "2023_09_15"
    eval_dataset = "AnisodiffOnCityscapes"
    # eval_dataset = "originalCityscapesVal"
    # evaluation of texture expert on original Cityscapes dataset:
    config=AttrDict({
            "num_runs": 1,
            "epochs": 200,
            "lr": 0.0005,
            # "model": "texture",
            "model": "anisodiff",
            # "model_class": "deeplabv3resnet50",
            "model_class": "deeplabv3resnet18",
            "input_channels": 3,
            "reset_weights": True,
            "num_classes": 19,
            "generate_train_ids_from_labels": False,
            # "generate_train_ids_from_labels": True,
            "dataset_name": 'CityscapesRgb19classes',
            # "dataset_name": 'CityscapesTexture2',
            # "dataset_name": 'CityscapesHEDBlackEdges19classes',
            "split": "val",
            # "split": "train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra",
            "transform_train": {"random_crop": (512, 512),
                                "to_tensor": True, #irrelevant. This is ignored by the code and toTensor is always done for images
                                # "normalization_mean": (0.2751, 0.3097, 0.2739),
                                # "normalization_var": (0.2093, 0.2146, 0.2105)},
                                "normalization_mean": [0.2544, 0.2949, 0.2515],
                                "normalization_var": [0.1803, 0.1854, 0.1826]},
            "dataset_test_root": "/home/datasets/Cityscapes",
            "dataset_test_label_root": "/home/datasets/Cityscapes",
            # "dataset_test_root": "/home/xxx/texture-voronoi-diagrams/Voronoi_2023_03_03_upsampled_texture_images",
            # "dataset_test_label_root": "/home/xxx/texture-voronoi-diagrams/Voronoi_2023_03_03_upsampled_texture_images_label",
            # "dataset_test_root": "/home/xxx/data/anisodiff/cityscapes/8192/leftImg8bit",
            # "dataset_test_label_root": "/home/xxx/data/train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra/pseudo_gt_fine",


            "transform_test_split": {"to_tensor": True, #irrelevant. This is ignored by the code and toTensor is always done for images
                                    # "normalization_mean": (0.2751, 0.3097, 0.2739),
                                    # "normalization_var": (0.2093, 0.2146, 0.2105)},
                                    "normalization_mean": [0.2544, 0.2949, 0.2515],
                                    "normalization_var": [0.1803, 0.1854, 0.1826]},
            "batch_size": 30,
            "batch_size_test": 10,
            # "batch_size_test": 4,
            # "num_workers": 10,
            "num_workers": 4,
            "weights": [],
            "lr_policy": "linear",
            "ckpt_path": f"experiments/{name}_{prod_date}/{name}_best_model.pth",
            "pred_root": f"experiments/{name}_{prod_date}",
            "exp_name": name,
            "dump_pred": True,
            "seed": 42,
            "log_freq": 2975,
        })


    testloader = get_dataloader(config.dataset_name,
                   config.dataset_test_root,
                   config.dataset_test_label_root,
                   config.transform_test_split,
                   split=config.split,
                   batch_size=config.batch_size_test,
                   num_workers=config.num_workers,
                   train_on_train_id=config.generate_train_ids_from_labels
                   )

    evaluate_expert(config, testloader, f"best", eval_dataset, log_images=True)
