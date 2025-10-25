
import argparse
import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm


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



def generate_contour_patched_texture_images(dataset_root_path, texture_patches_dir, texture_id=0, final_img_shape=(1024, 2048, 3), split='val', target_type="labelTrainIds", rnd_seed=42):
    random.seed(rnd_seed)

    final_texture_img_path = os.path.join(f"{texture_patches_dir}_in_contour_{texture_id}", split)
    os.makedirs(final_texture_img_path, exist_ok=True)

    filled_img = Image.fromarray(np.zeros(final_img_shape, np.uint8))

    img_dir = os.path.join(dataset_root_path, split)
    for root, _, fnames in sorted(os.walk(img_dir)):
        for fname in tqdm(fnames, "Loading images"):
            if is_image_file(fname):
                root_parts = root.split('leftImg8bit')
                label_path = os.path.join(root_parts[0].rstrip('/'), 'gtFine', root_parts[1].lstrip('/'),
                                          f"{fname.split('_leftImg8bit')[0]}_gtFine_{target_type}.png")
            else:
                continue
            texture_dir = os.path.join(f"{texture_patches_dir}_{texture_id}", split)
            np_label = np.asarray(Image.open(label_path))
            if (np_label.shape[0] != final_img_shape[0]) or (np_label.shape[1] != final_img_shape[1]):
                exit(f'Label image and final image need to have the same width and height')
            for label_id in np.unique(np_label):
                for texture_root, _, texture_fnames in sorted(os.walk(texture_dir)):

                    # randomly choose texture img
                    rnd_texture_img_file = random.choice([f for f in texture_fnames if f.startswith('final')])
                    rnd_texture_img_path = os.path.join(texture_root, rnd_texture_img_file)
                    texture_img = Image.open(rnd_texture_img_path)


                    # Mask all but selected class
                    mask = np.zeros(np_label.shape, dtype=np.uint8)
                    mask[np_label == label_id] = 255

                    # fill segments with texture image
                    filled_img.paste(texture_img, mask=Image.fromarray(mask, mode="L"))

            filled_img.save(os.path.join(final_texture_img_path,
                            f"texture_contour_patched_{texture_id}_{fname.split('_leftImg8bit')[0]}.png"))
            filled_img = Image.fromarray(np.zeros(final_img_shape, np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Generate contour filled texture images',
        description='crops class specific texture is croped from mosaic images and filled in a given segmentation mask')
    parser.add_argument('--label_id',
                        type=int,
                        default=15,
                        help='class id to generate crops and mosaics from')
    parser.add_argument('--dataset_root_path',
                        default=os.path.join('datasets', 'Cityscapes', 'leftImg8bit'),
                        help='root path to dataset RGB images e.g. Cityscapes')
    parser.add_argument('--texture_root',
                        default="2024_07_15_upsampled_texture_images",
                        help='Path to already produced texture images')
    parser.add_argument('--split',
                        default="train",
                        help='Split of your dataset.')

    args = parser.parse_args()
    dataset_root_path = args.dataset_root_path
    texture_root = args.texture_root
    label_id = args.label_id
    split_data = args.split
    
    generate_contour_patched_texture_images(dataset_root_path, texture_root, label_id, (1024, 2048, 3), split='val',
                             target_type='labelTrainIds', rnd_seed=42)

# end main

