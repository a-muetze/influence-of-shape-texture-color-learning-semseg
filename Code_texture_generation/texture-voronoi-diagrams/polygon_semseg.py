
import argparse
import cv2
import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

import albumentations



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    '''
    Checks if a given file has a proper imagefile extention and therefore is most likely an image file.
    This code is extracted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/image_folder.py
    '''
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def generate_texture_images(dataset_root_path,
                            result_path,
                            label_id=13,
                            final_img_shape=(1024, 2048, 3),
                            split='val',
                            target_type="labelTrainIds",
                            rnd_seed=42,
                            upsampling_factor=1):
    '''
    Generates texture mosaic images of a given label_id of a given base dataset with segmentation masks
        
    
    '''
    random.seed(rnd_seed)
    # augementation settings
    # Declare an augmentation pipeline
    aug_list = [albumentations.HorizontalFlip(p=1),
                albumentations.RandomCropFromBorders(0.3, 0.3, 0.3, 0.3, p=1),
                albumentations.ShiftScaleRotate(scale_limit=0.25,
                                                rotate_limit=12,
                                                p=0.75,
                                                border_mode=cv2.BORDER_CONSTANT,
                                                value=0),

                ]

    rgb_crops = []
    pad_y = 10
    pad_x = 10
    padded_img_shape = (final_img_shape[0] + 2*pad_y, final_img_shape[1] + 2*pad_x, final_img_shape[2])
    slice_y = slice(pad_y, final_img_shape[0] + pad_y)
    slice_x = slice(pad_x, final_img_shape[1] + pad_x)
    img_slice = (slice_y, slice_x, slice(None))

    count_full = 0
    texture_img_path = os.path.join(f"{result_path}_{label_id}", split)
    os.makedirs(texture_img_path, exist_ok=True)
    zero_padded = np.zeros(padded_img_shape, dtype=np.uint8)
    filled_img = Image.fromarray(np.zeros(padded_img_shape, np.uint8))

    black = Image.fromarray(np.zeros(padded_img_shape, np.uint8))
    filled_mask = Image.fromarray(np.zeros(padded_img_shape, np.uint8))
    # Load image and segmenation mask 
    img_dir = os.path.join(dataset_root_path, split)
    for root, _, fnames in sorted(os.walk(img_dir)):
        #j = 0
        for fname in tqdm(fnames, "Loading images"):
            if is_image_file(fname):
                img_path = os.path.join(root, fname)
                root_parts = root.split('leftImg8bit')
                label_path = os.path.join(root_parts[0].rstrip('/'), 'gtFine', root_parts[1].lstrip('/'),
                                          f"{fname.split('_leftImg8bit')[0]}_gtFine_{target_type}.png")
                # label_path = os.path.join(root_parts[0].rstrip('/'), 'pseudo_gt_fine', root_parts[1].lstrip('/'),
                #                           f"{fname}.png")
            else:
                continue
           
            np_label = cv2.imread(label_path)
            rgb_img = cv2.imread(img_path)
            # generate class-specif mask (class segments white, rest black)
            mask = np.zeros(np.shape(np_label), dtype=np.uint8)
            mask[np_label == label_id] = 255

            # Mask original image with the generated mask
            rgb_polygons_img = cv2.bitwise_and(mask, rgb_img)

            # Saving for visualization and debug reasons
            # mask_path = os.path.join(f"mask_images_{label_id}", split)
            # os.makedirs(mask_path, exist_ok=True)
            # cv2.imwrite(os.path.join(mask_path, f"{fname}"), mask

            # polygon_path = os.path.join(f"polygon_images_{label_id}", split)
            # os.makedirs(polygon_path, exist_ok=True)
            # cv2.imwrite(os.path.join(polygon_path, f"{fname}"), rgb_polygons_img)

            # polygon_crops_path = os.path.join(f"polygon_images_{label_id}", split, "crops")
            # os.makedirs(polygon_crops_path, exist_ok=True)

            # Find connected components within one class-based masked image
            mask_graysc = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(mask_graysc, 127, 255, cv2.THRESH_BINARY)
            # Find object contours
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if contours == None:
                break
            big_mask_single_contour = np.zeros(np.shape(rgb_img), np.uint8)
            tmp = np.zeros(np.shape(big_mask_single_contour))
            for i, contour in enumerate(contours):
                # inverse version of this https://stackoverflow.com/questions/59577466/building-nested-mask-from-contours-with-opencv serves as idea for this section
                #outer contour
                if (hierarchy[0][i][3]<0):
                    cv2.drawContours(tmp, contours, i,(255, 255, 255), -1)
            for i, contour in enumerate(contours):
                #Inner contour
                if (hierarchy[0][i][2]<0 and hierarchy[0][i][3]>-1):
                    cv2.drawContours(big_mask_single_contour, contours, i,(255, 255, 255), -1)
            tmp =  tmp - big_mask_single_contour
            big_mask_single_contour =  tmp
            # Problem: which CC needs to be handled with hirachy and which without? https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
            for i, contour in enumerate(contours):
                if (hierarchy[0][i][2]<0 and hierarchy[0][i][3]>-1):
                    continue 
                x, y, w, h = cv2.boundingRect(contour)
                x_min = x
                y_min = y
                # remove to small components
                if cv2.contourArea(contour) < 36:
                    continue

                # crop image
                rgb_crop = rgb_polygons_img[y_min:(y_min + h), x_min:(x_min + w)]
                mask_crop_single_contour = big_mask_single_contour[y_min:(y_min + h), x_min:(x_min + w)]
                cont_area = cv2.contourArea(contour)
                sampling_id = 0
                # transform crop to enlarge data pool
                while sampling_id < upsampling_factor:
                    rgb_crops.append({"crop": rgb_crop, "area": cont_area, "width": w, "hight": h, "mask": mask_crop_single_contour}) # add "mask": mask
                    sampling_id += 1
                    if sampling_id > upsampling_factor:
                        break
                    for transform in aug_list:
                        augmented_data = {"image": rgb_crop, "mask": mask_crop_single_contour}
                        transformation = transform(**augmented_data)
                        augmented_crop, augmented_mask = transformation["image"], transformation["mask"]
                        rgb_crops.append({"crop": augmented_crop,
                                        "area": np.NAN,
                                        "width": augmented_crop.shape[1],
                                        "hight": augmented_crop.shape[0],
                                        "mask": augmented_mask})
                        sampling_id += 1
                        if sampling_id > upsampling_factor:
                            break
            

    # combine crops to a fully filled texture mosaic image
    if rgb_crops:
        # shuffle crops for diversity
        random.shuffle(rgb_crops)

        last_row = 0
        last_col = 0
        crop_id = 0
        # start with a black mask
        filled_mask = np.zeros((padded_img_shape[0], padded_img_shape[1]), dtype=np.uint8)
        np_black = np.array(black) # helper mask to mask already filled areas in the mosaic image
        filled_mask[np.squeeze((np_black == zero_padded).all(axis=2, keepdims=True), axis=2)] = 255
        for crop_dict in tqdm(rgb_crops, "Adding crops"):
            use_biggest_patch = (random.random() < 0.1)
            if use_biggest_patch:
                empty_patches, _ = cv2.findContours(
                    filled_mask[(slice_y, slice_x)], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                if contours == None:
                    break
                sorted_empty_patches = sorted(empty_patches, reverse=True,
                                              key=lambda contour: cv2.contourArea(contour))
                if len(sorted_empty_patches) > 10:
                    x, y, w, h = cv2.boundingRect(sorted_empty_patches[0])
                    most_empty_row = x
                    most_empty_col = y
                else:
                    most_empty_row = np.argmax(
                        (cv2.reduce(filled_mask[(slice_y, slice_x)], 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1))/255)
                    first_empty_col = np.where(filled_mask[most_empty_row + pad_x, slice_x] == 255)
                    most_empty_col = first_empty_col[0][0]

            #  find most empty row in the mask of the so far filled image (ignoring padding) and the first empty column within that row
            #  use index with padding to avoid empty areas near the image boundary
            else:
                most_empty_row = np.argmax(
                    (cv2.reduce(filled_mask[(slice_y, slice_x)], 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1))/255)
                first_empty_col = np.where(filled_mask[most_empty_row + pad_x, slice_x] == 255)
                most_empty_col = first_empty_col[0][0]

            # if insertion position has not changed since last time, noise is added to the insertion coordinates
            if (most_empty_col == last_col) and (most_empty_row == last_row):
                shifted_most_empty_col = most_empty_col + \
                    random.randint(-crop_dict["crop"].shape[0], crop_dict["crop"].shape[0])

                most_empty_col = (shifted_most_empty_col if shifted_most_empty_col > 0 else 0)
                most_empty_col = (most_empty_col if shifted_most_empty_col <
                                  padded_img_shape[0] else padded_img_shape[0]-1)

                shifted_most_empty_row = most_empty_row + \
                    random.randint(-crop_dict["crop"].shape[0], crop_dict["crop"].shape[0])
                most_empty_row = (shifted_most_empty_row if (shifted_most_empty_row > 0) else 0)
                most_empty_row = (most_empty_row if (shifted_most_empty_row <
                                                     padded_img_shape[1]) else padded_img_shape[1]-1)

            last_row = most_empty_row
            last_col = most_empty_col

            # add crop in the biggest black hole
            crop = Image.fromarray(zero_padded.copy())
            crop.paste(Image.fromarray(crop_dict["crop"]), (most_empty_col, most_empty_row))
            filled_img.paste(crop, (0, 0), Image.fromarray(filled_mask))
        
            m_crop = Image.fromarray(zero_padded.copy())
            m_crop.paste(Image.fromarray(crop_dict['mask'].astype(np.uint8)), (most_empty_col, most_empty_row))
            black.paste(m_crop, (0,0), Image.fromarray(filled_mask))
            # get non padded image converted from bgr to rgb
            originally_sized_rgb_array = np.array(filled_img)[img_slice][:, :, ::-1]

            # save it after each 100 crops
            if crop_id % 99 == 0:
                save_img = Image.fromarray(originally_sized_rgb_array)
                save_img.save(os.path.join(texture_img_path, f"tmp{label_id}_{count_full}.png"))

            # New image if mosaic images is completely filled
            filled_mask = np.zeros((padded_img_shape[0], padded_img_shape[1]), dtype=np.uint8)
            np_black  = np.array(black)
            filled_mask[np.squeeze((np_black == zero_padded).all(axis=2, keepdims=True), axis=2)] = 255 

            if (filled_mask[(slice_y, slice_x)]!=255).all():
                save_img = Image.fromarray(originally_sized_rgb_array)
                save_img.save(os.path.join(texture_img_path, f"final{label_id}_{count_full}.png"))
                filled_img = Image.fromarray(np.zeros(padded_img_shape, np.uint8))

                filled_mask = np.zeros((padded_img_shape[0], padded_img_shape[1]), dtype=np.uint8)
    
                black = Image.fromarray(np.zeros(padded_img_shape, np.uint8))
                np_black = np.array(black)
                filled_mask[np.squeeze((np_black == zero_padded).all(axis=2, keepdims=True), axis=2)] = 255

                
                # delete tmp file
                tmp_file = os.path.join(texture_img_path, f"tmp{label_id}_{count_full}.png")
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                else:
                    print(f"The file '{tmp_file}' does not exist")

                # count complete mosaic images
                count_full += 1
            # count processed crops
            crop_id += 1
    print(f"Label {label_id} was processed")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Generate texture mosaics',
        description='crops class specific texture from images and combines them randomly in a mosaic')
    parser.add_argument('--label_id',
                        type=int,
                        default=15,
                        help='class id to generate crops and mosaics from')
    parser.add_argument('--dataset_root_path',
                        default=os.path.join('datasets', 'Cityscapes', 'leftImg8bit'),
                        help='root path to dataset RGB images e.g. Cityscapes')
    parser.add_argument('--result_path',
                        default="upsampled_texture_images",
                        help='Result root path for texture mosaic images')
    parser.add_argument('--split',
                        default="train",
                        help='Split of your dataset.')

    args = parser.parse_args()
    dataset_root_path = args.dataset_root_path
    result_path = args.result_path
    label_id = args.label_id
    #cityscapes upscaling
    ups_factor_list = [4, 8, 2, 40, 32, 53, 160, 40, 4, 32, 9, 32, 240, 6, 120, 160, 160, 280, 80]
    #carla upsampling
    split_data = args.split
    # if split_data == 'test':
    #     ups_factor_list = [1, 1, 1, 5, 8, 24, 280, 1, 15, 0.5, 56, 3, 9, 12, 15, 35, 3, 280, 280, 20]
 
    # elif split_data == 'train':
    #     ups_factor_list = [1, 3, 1, 5, 5, 67, 120, 1, 6, 0.5, 43, 3, 12, 29, 25, 50, 3, 600, 28, 8]
    
    # else:
    #     print('Only "test" and "train" split are implemented. Please choose one of them.')
    # ups_factor_dict = {
    #     "1":    ups_factor_list[0],
    #     "2":    ups_factor_list[1],
    #     "3":    ups_factor_list[2],
    #     "4":    ups_factor_list[3],
    #     "6":    ups_factor_list[4],
    #     "7":    ups_factor_list[5],
    #     "8":    ups_factor_list[6],
    #     "9":    ups_factor_list[7],
    #     "10":   ups_factor_list[8],
    #     "11":   ups_factor_list[9],
    #     "12":   ups_factor_list[10],
    #     "14":   ups_factor_list[11],
    #     "15":   ups_factor_list[12],
    #     "16":   ups_factor_list[13],
    #     "20":   ups_factor_list[14],
    #     "21":   ups_factor_list[15],
    #     "22":   ups_factor_list[16],
    #     "23":   ups_factor_list[17],
    #     "25":   ups_factor_list[18],
    #     "28":   ups_factor_list[19]
    # }
    # #ups_factor_list = [4, 8, 2, 40, 32, 53, 160, 40, 4, 32, 9, 32, 240, 6, 120, 160, 160, 280, 80] # adapt

    generate_texture_images(dataset_root_path,
                            result_path,
                            label_id,
                            (1024, 2048, 3),
                            split=split_data,
                            target_type='labelTrainIds',
                            rnd_seed=42,
                            # # for Carla
                            # upsampling_factor=ups_factor_dict[f"{label_id}"])
                            # # for ciryscapes:
                            upsampling_factor=ups_factor_list[label_id])

# end main
