import os
import torch
from tqdm import tqdm
import yaml
from utils.evaluate_utils import dump_prediction_mask, replace_jpg_with_png
from PIL import Image

from utils.training_utils import find_model_using_name, get_dataloader, load_checkpoint
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == "__main__":

    # name = "contour_hed_blackedge_CS19_deeplabv3resnet18_512x512_pseudoVal"
    # prod_date = "2023_07_31"
    name = "texture_gray_CS19_deeplabv3resnet18_512x512_Adam_cs_extra_train_val_3aaeckjd"
    prod_date = "2023_09_02"
    # name = "texture_colored_CS19_deeplabv3resnet18_512x512_cs_extra_train_val_36uw431c"
    # prod_date = "2023_09_06"
    # name = "contour_anisodiff_CS19_deeplabv3resnet18_512x512_pseudoVal_1gpag6aw"
    # prod_date = "2023_09_15"
    # name = "Color_HSmanuell_CS19_LR-4_conv1x1_stack256_256_Custom_FCNHead_3fn6qt1r"
    # prod_date = "2023_10_28"


    config={
        # "model": "contour",
        "model": "texture",
        # "model": "anisodiff",
        # "model": "color",
        "model_class": "deeplabv3resnet18",
        "input_channels": 1,
        # "input_channels": 2,
        # "input_channels": 3,
        "reset_weights": True,
        "num_classes": 19,
        "generate_train_ids_from_labels": True,
        ## "generate_train_ids_from_split_labels": True,
        ## "generate_train_ids_from_train_labels": False,
        # "generate_train_ids_from_labels": False,
        # "dataset_name": 'CityscapesHEDBlackEdges19classes',
        "dataset_name": 'CityscapesRgb19classes',
        # "dataset_name": 'CityscapesHSV19classes',
        "split": "train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra",  # split='train_testsplit' or 'val or 'train'
        # "split": "train",  # split='train_testsplit' or 'val or 'train'

        # "dataset_test_root": "/home/xxx/data/cityscapes_hed_black_edges",
        # "dataset_test_root": "/home/datasets/Cityscapes",
        "dataset_test_root": "/home/xxx/data/",
        # "dataset_test_label_root": "/home/datasets/Cityscapes",
        "dataset_test_label_root": "/home/xxx/data/",
        #colored texture wo contour
        # "transform_test_split": {"to_tensor": True,
        #                          "normalization_mean": (0.2751, 0.3097, 0.2739),
        #                          "normalization_var": (0.2093, 0.2146, 0.2105)},

        # # HED (contour_hed_blackedge_CS19_deeplabv3resnet18_512x512_pseudoVal)
        # "transform_test_split": {"to_tensor": True,
        #                          "normalization_mean": 0.9037,
        #                          "normalization_var":  0.18},

        # colored texture in contour (texture_colored_CS19_deeplabv3resnet18_512x512_cs_extra_train_val_36uw431c)
        # "transform_test_split": {"to_tensor": True,
        #                          "normalization_mean": [0.2753, 0.3099, 0.274],
        #                          "normalization_var": [0.2081, 0.2134, 0.2091]},

        # # gray texture in contour
        "transform_test_split": {"to_tensor": True,
                                 "normalization_mean": 0.2955,
                                 "normalization_var": 0.2102},

        # # Anisodiff (contour_anisodiff_CS19_deeplabv3resnet18_512x512_pseudoVal_1gpag6aw)
        # "transform_test_split": {"to_tensor": True, # ignored but is always applied for images
        #                         "normalization_mean": [0.2544, 0.2949, 0.2515],
        #                         "normalization_var": [0.1803, 0.1854, 0.1826]},

        # # Color HS(V) manuell
        # "transform_test_split": {"to_tensor": True,
        #                          "normalization_mean": [0.3287114, 0.1922254],
        #                          "normalization_var": [0.0873381, 0.0792537]},

        "batch_size_test": 4,
        "drop_last": False,
        "num_workers": 4,
        "ckpt_path": f"experiments/{name}_{prod_date}/{name}_best_model.pth",
        "pred_root": f"experiments/{name}_{prod_date}/",
        "exp_name": name,
        "dump_pred": False,
        "seed": 42,
        "optim": "Adam",
        "save_softmax": True,
        "save_softmax_path": f"data/softmax/{name}_{prod_date}",
        # "source_root": "/home/xxx/data/train_testsplit_cityscapes_val_randomly_sampled_from_sc_train_extra/leftImg8bit",
        "source_root": '/home/datasets/Cityscapes',
        # "data_types": ["hed", "rgb"],
        "data_types": ["rgb"],
        "blending_alpha": 0.5
    }

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_dl = get_dataloader(config["dataset_name"],
                            config["dataset_test_root"],
                            config["dataset_test_label_root"],
                            # split='train_testsplit' or 'val
                            split=config["split"],
                            transforms=config["transform_test_split"],
                            batch_size=config["batch_size_test"],
                            num_workers=config["num_workers"],
                            train_on_train_id=config["generate_train_ids_from_labels"],
                            drop_last=config["drop_last"]
                            )
    model_obj = find_model_using_name(config["model_class"])
    model = model_obj(AttrDict(config), config["input_channels"])
    if config["ckpt_path"] != None:
        checkpoint = torch.load(config["ckpt_path"], map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        epoch = checkpoint['epoch']
    model.to(device)
    model.eval()


    # set up directory IO
    save_dir_blended = []
    if config["save_softmax"]:
        save_softmax_dir = os.path.join(config["save_softmax_path"],
                                        str(epoch),
                                        os.path.basename(config["dataset_test_root"]),
                                        config["split"],
                                        )
        if not os.path.exists(save_softmax_dir):
            os.makedirs(save_softmax_dir)
            os.makedirs(os.path.join(save_softmax_dir, "configs"))
            print(f"Created: {save_softmax_dir} to store softmax outputs")
        # Save config as yaml in softmax folder
        with open(os.path.join(save_softmax_dir, "configs", f"config_eval_{config['exp_name']}.yaml"), "w") as fh:
            yaml.dump(config, fh)

    if config["dump_pred"]:
        save_config_dir = os.path.join(config["pred_root"],
                                os.path.basename(config["dataset_test_root"]),
                                config["split"],
                                "configs",
                                str(epoch))
        if not os.path.exists(save_config_dir):
            os.makedirs(save_config_dir)
            print(f"Created: {save_config_dir} to store eavluation configs")

        # Save config as yaml in prediction folder
        with open(os.path.join(save_config_dir, f"config_eval_{config['exp_name']}.yaml"), "w") as fh:
            yaml.dump(config, fh)

        for data_type in config["data_types"]:
            save_dir = os.path.join(config["pred_root"],
                                    os.path.basename(config["dataset_test_root"]),
                                    config["split"],
                                    f"blended",
                                    str(epoch),
                                    data_type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"Created: {save_dir} to store blended images")
            save_dir_blended.append(save_dir)


    # Infer images
    with torch.inference_mode():
        # loop through validation set:
        for images, labels, paths in tqdm(valid_dl, desc="Inferring images", unit_scale=config["batch_size_test"]):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)['out']
            logits = outputs.data.cpu()

            if config["save_softmax"]:
                softmax = torch.nn.functional.softmax(logits,dim=1)
                softmax_list = torch.split(softmax, 1, dim=0)
                for tensor_split, path_i in zip(softmax_list, paths):
                    save_path = os.path.join(save_softmax_dir, f"{os.path.basename(path_i).removesuffix('.png')}.pt")
                    with open( save_path, 'wb') as fileObj:
                        softmax_split = torch.squeeze(tensor_split).to(torch.float16)
                        torch.save(softmax_split, fileObj)

            if not config["dump_pred"]:
                continue

            pred = torch.squeeze(torch.argmax(logits, dim=1))
            pred_array = pred.numpy()

            color_palette = valid_dl.dataset.color_palette_train_ids

            for pred_i, path_i in zip(pred_array, paths):
                colored_pred = dump_prediction_mask(pred_i,
                                                    os.path.join(config["pred_root"],
                                                                 os.path.basename(config["dataset_test_root"]),
                                                                 config["split"]),
                                                    color_palette,
                                                    epoch,
                                                    os.path.basename(
                                                        replace_jpg_with_png(path_i)
                                                        )
                                                    )
                img = Image.open(path_i)
                blend = Image.blend(img.convert("RGBA"), colored_pred.convert("RGBA"), config["blending_alpha"])
                blend.save(os.path.join(save_dir_blended[0], os.path.basename(path_i)))
                if len(config["data_types"]) > 1 :
                    if os.path.basename(config["source_root"]).lower() == "cityscapes":
                        city = os.path.basename(path_i).split("_")[0]
                        img_path = os.path.join(config["source_root"], "leftImg8bit", config["split"], city, os.path.basename(path_i))
                    else:
                        img_path = os.path.join(config["source_root"], os.path.basename(path_i))
                    img_rgb = Image.open(img_path)
                    blend = Image.blend(img_rgb.convert("RGBA"), colored_pred.convert("RGBA"), config["blending_alpha"])
                    blend.save(os.path.join(save_dir_blended[1], os.path.basename(path_i)))


# end main
