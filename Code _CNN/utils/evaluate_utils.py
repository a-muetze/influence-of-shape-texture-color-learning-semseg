import os
from PIL import Image
from matplotlib import pyplot as plt

def dump_prediction_mask(pred,
                         pred_root,
                         color_mapping,
                         epoch=None,
                         iter=None,
                         dump_type=f"colored"):
    # gather info for saving prediction
    #file_name = os.path.basename(image_path)
    pred_save_path = os.path.join(pred_root,
                                  dump_type,
                                  str(epoch),
                                  f'{str(iter)}')
    save_dir = os.path.dirname(pred_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created:", save_dir)

    # colorize prediction
    color_palette = color_mapping
    colored_pred = Image.fromarray(pred.astype("uint8")).convert('P')
    colored_pred.putpalette(color_palette)
    colored_pred.save(pred_save_path)
    return colored_pred
    #print("Prediction saved:", pred_save_path)


def dump_heatmap_mask(weight,
                      pred_root,
                      epoch=None,
                      iter=None,
                      dump_type=f"colored"):
    # gather info for saving heatmap
    #file_name = os.path.basename(image_path)
    pred_save_path = os.path.join(pred_root,
                                  dump_type,
                                  str(epoch),
                                  f'{str(iter)}')
    save_dir = os.path.dirname(pred_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Created:", save_dir)

    # colorize heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(weight, cmap='PuBuGn', interpolation='nearest', vmin=0, vmax=1)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label='weight of experts prediciton', shrink=.5)
    fig.tight_layout()
    plt.savefig(pred_save_path, transparent=True)
    plt.close()


def replace_jpg_with_png(file_path):
    # Check if the file path ends with '.jpg' (case-insensitive)
    if file_path.lower().endswith('.jpg'):
        # Replace '.jpg' with '.png'
        new_file_path = file_path[:-4] + '.png'
        return new_file_path
    else:
        # Return the original file path if it doesn't end with '.jpg'
        return file_path
