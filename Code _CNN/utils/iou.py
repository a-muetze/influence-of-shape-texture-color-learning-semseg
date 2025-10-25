#!/usr/bin/python
'''
written by XXX
adapted by XXX
'''

import itertools
import os
from random import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import math
import sys

from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

def print_error(message):
    """Print an error message and quit"""
    print('\n-----\nERROR: ' + str(message) + "\n...good bye...")
    sys.exit(-1)


class Style:
    """Class for colors"""
    RED = '\033[31;1m'
    GREEN = '\033[32;1m'
    YELLOW = '\033[33;1m'
    BLUE = '\033[34;1m'
    MAGENTA = '\033[35;1m'
    CYAN = '\033[36;1m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def get_color_entry(value):
    """Colored value output if colorized flag is activated."""
    if not isinstance(value, float) or math.isnan(value):
        return Style.ENDC
    if value < .20:
        return Style.RED
    elif value < .40:
        return Style.YELLOW
    elif value < .60:
        return Style.BLUE
    elif value < .80:
        return Style.CYAN
    else:
        return Style.GREEN


def generate_matrix(num_classes):
    """Generate empty confusion matrix"""
    max_id = num_classes
    return np.zeros(shape=(max_id, max_id), dtype=np.ulonglong)  # longlong for no overflows


def get_iou_score_for_label(label, conf_matrix):
    """Calculate and return IoU score for a particular label"""
    tp = np.longlong(conf_matrix[label, label])
    fn = np.longlong(conf_matrix[label, :].sum()) - tp
    notIgnored = [l for l in range(len(conf_matrix)) if not l == label]
    fp = np.longlong(conf_matrix[notIgnored, label].sum())
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom


def get_score_average(score_list):
    """Get average of scores, only computes the average over valid entries"""
    validScores = 0
    scoreSum = 0.0
    for score in score_list:
        if not math.isnan(score_list[score]):
            validScores += 1
            scoreSum += score_list[score]
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores



def print_class_scores(score_list, class_names, fh):
    """Print intersection-over-union scores for all classes"""
    fh.write("classes          IoU      " "\n")
    fh.write("--------------------------------\n")
    print(Style.BOLD + "classes          IoU      " + Style.ENDC)
    print("--------------------------------")
    for label in range(len(class_names)):
        labelName = str(class_names[label])
        iouStr = get_color_entry(score_list[labelName]) + "{val:>5.4f}".format(val=score_list[labelName]) + Style.ENDC
        print("{:<14}: ".format(labelName) + iouStr + "    ")
        fh.write("{:<14}: ".format(labelName) + "{val:>5.4f}".format(val=score_list[labelName]) + "    \n")


def evaluate_img_lists(pred_list, gt_list, num_classes, ignore_in_eval_ids, class_names, epoch, fh):
    """Evaluate image lists pairwise"""
    if len(pred_list) != len(gt_list):
        print_error("List of images for prediction and ground truth are not of equal size.")
    confusion_matrix = generate_matrix(num_classes)

    # Evaluate all pairs of images and save them into a matrix
    for i in range(len(pred_list)):
        # Loading all resources for evaluation and check for errors.
        pred, gt = None, None
        gt = np.array(gt_list[i])
        pred = np.array(pred_list[i])
        for pred_j, gt_j in zip(pred, gt):
            confusion_matrix = evaluate_pair(pred_j, gt_j, confusion_matrix, ignore_in_eval_ids)
        print("\rImages Processed: {}".format(pred.shape[0]* (i + 1)), end=' ')
        sys.stdout.flush()

    classScoreList = {}
    for label in range(num_classes):
        labelName = class_names[label]
        classScoreList[labelName] = get_iou_score_for_label(label, confusion_matrix)
    print("\n")
    print_class_scores(classScoreList, class_names, fh)
    iouAvgStr = get_color_entry(get_score_average(classScoreList)) + "{avg:5.3f}".format(
        avg=get_score_average(classScoreList)) + Style.ENDC

    fh.write("--------------------------------")
    fh.write("Score Average : " + "{avg:5.3f}".format(
        avg=get_score_average(classScoreList)) + " (mIoU)")
    fh.write("--------------------------------")
    fh.write("")
    print("--------------------------------")
    print("Score Average : " + iouAvgStr + " (mIoU)")
    print("--------------------------------")
    print("")
    plot_confusion_matrix(confusion_matrix, class_names, epoch)

    return get_score_average(classScoreList)


def evaluate_pair(pred, gt, conf_matrix=None, ignore_in_eval_ids=None):
    """
    Main evaluation method. Evaluates pairs of prediction and ground truth with target type 'semantic_train_id',
    then updates confusion matrix
    """

    if ignore_in_eval_ids is not None:
        pred = pred[~np.isin(gt, ignore_in_eval_ids)]
        gt = gt[~np.isin(gt, ignore_in_eval_ids)]
    try:
        encoding_value = max(np.max(gt), np.max(pred)).astype(np.int32) + 1
    except ValueError:
        print(f"pred and or gt are empty")
        return conf_matrix
    encoded = (gt.astype(np.int32) * encoding_value) + pred
    values, counts = np.unique(encoded, return_counts=True)
    if conf_matrix is None:
        conf_matrix = np.zeros((encoding_value, encoding_value))
    for value, c in zip(values, counts):
        pred_id = value % encoding_value
        gt_id = int((value - pred_id) / encoding_value)
        conf_matrix[gt_id][pred_id] += c
    return conf_matrix



def plot_confusion_matrix(cm, class_names, epoch, save_path=None, fontsize=33, fontsize2=25):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20
})

    font_files = font_manager.findSystemFonts(fontpaths='utils/fonts')
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    fig, ax = plt.subplots(figsize=(20, 20))
    cm_normalized = np.true_divide(cm, cm.sum(axis=1, keepdims=True))
    cm_normalized = np.ma.masked_invalid(cm_normalized)
    heat = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.PuBuGn) ##plasma magma YlGn YlGnBu

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names.values(), rotation=90, fontsize=fontsize, fontdict={
                'fontweight': plt.rcParams['axes.titleweight']}) #rotation=45
    plt.yticks(tick_marks, class_names.values(), fontsize=fontsize)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)


    # Use white text if squares are dark; otherwise black.
    threshold = (cm_normalized.max() + cm_normalized.min()) * 0.85
    for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm.shape[1])):
        # color = "white" if (cm_normalized[i, j] < threshold and ~cm_normalized.mask[i, j]) else "black"
        color = "white" if (cm_normalized[i, j] > threshold and ~cm_normalized.mask[i, j]) else "black"
        # color = "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", verticalalignment="center",
                 color=color, fontsize=fontsize2)#, fontfamily='Vollkorn SC', weight='bold' )


    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize, labelpad=15)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)
    # ax.tick_params(labelsize=fontsize-2)
    cax.tick_params(labelsize=fontsize-2)

    fig.colorbar(heat, cax=cax)
    plt.tight_layout()
    if not save_path:
        plt.savefig(f'confusion_{epoch}.png')
    else:
        plt.savefig(os.path.join(save_path, f'confusion_{epoch}.pdf'))
    return fig


if __name__ == "__main__":
    classes = {"1":"cl1", "2":"cl2", "3":"cl3", "4":"cl4", "5":"cl5", "6":"cl6", "7":"cl7"}
    confu = np.random.rand(7,7)*1000
    confu = np.zeros((7,7))
    pred = np.random.randint(0,7,1000)
    gt = np.random.randint(0,7,1000)
    print(pred)
    print(gt)
    confu2 = evaluate_pair(pred,gt,confu)
    #conf_normalized = np.true_divide(confu, confu.sum(axis=1, keepdims=True))
    print(confu2)
    fig = plot_confusion_matrix(confu2, classes, 'debug')
    #plot_conf_matrix(confu2, classes)
