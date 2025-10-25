import os
from time import time
import numpy as np

import torch
from tqdm import tqdm


def calc_dataset_mean_std(dataset, channels):
    torch.set_printoptions(sci_mode=False, precision=7)

    N_CHANNELS = channels
    full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count()//8)

    before = time()
    mean = torch.zeros(N_CHANNELS)
    std = torch.zeros(N_CHANNELS)
    print('==> Computing mean and std..')
    for inputs, _labels, _paths in tqdm(full_loader):
        for i in range(N_CHANNELS):
            mean[i] += inputs[:, i, :, :].float().mean()
            std[i] += inputs[:, i, :, :].float().std()
    mean.div_(len(dataset))
    std.div_(len(dataset))

    print("time elapsed: ", time()-before)

    print(f"{dataset.__class__.__name__}:\n mean:{mean}\n std:{std}")

    torch.set_printoptions(profile='default')


def calc_class_distribution(gt_image_path, target, num_classes, verbose=False, strange_channel=False):
    # total_label = np.asarray(imageio.imread(gt_image_path, format='PNG-FI'))
    # if strange_channel:
    #     seg_label = np.uint8(np.array(total_label[:,:,0]))
    # else:
    #     seg_label = total_label
    seg_label = target
    # if dataset.split == "train":
    #     # transform segmentation mask according to train_ids
    #     for label in dataset.labels:
    #         mask = [seg_label == label.id]
    #         seg_label[tuple(mask)] = label.train_id
    class_dist = np.histogram(seg_label, bins=np.append(np.arange(0, num_classes), [255, 256]))
    class_dens = np.histogram(seg_label, bins=np.append(np.arange(0, num_classes), [255, 256]), density=True) # +1 to ensure each class is in different bin, as the last bin includes both boundaries
    if verbose:
        print('\n')
        print(class_dens[0])
        print('\n\n')
        print(class_dist[0])

    return class_dist[0]

def calc_datset_class_dist(dataset, num_classes, verbose=False, strange_channel=False):
    total_dist = np.zeros(num_classes+1)
    amount_of_gt = 0
    gt_image_path = ""
    for data, target, path in dataset:
    #for gt_image_path in dataset.targets:
            total_dist += calc_class_distribution(gt_image_path,
                                                  target,
                                                  num_classes,
                                                  verbose=verbose,
                                                  strange_channel=strange_channel)
            amount_of_gt += 1
    mean_dist = total_dist/amount_of_gt
    return mean_dist, total_dist
