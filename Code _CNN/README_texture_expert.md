# Texture based experts


## Dataset preparation
Generate TEXTURE data from your data.
Split data into train, test, val.

### Dataset
* write a dataloader for the data
* calculate mean and std for your custum dataset train split with `calc_dataset_mean_std.py` which uses the `datasets/mean_std.py` script.

## Train texture expert
* If you don't have a wandb account disable wandb by typing `wandb disabled` in the terminal you want to start the script
* set parameters in `main_texture.py`
* most important parameters:
    * exp_name
    * dataset type
    * dataset train and val paths
    * normalization parameters

## Evaluation
The expert can be evaluated by running the `main_evaluate_experts.py`
