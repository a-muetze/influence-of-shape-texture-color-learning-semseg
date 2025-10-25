# Shape based experts


## Dataset preparation
Select one expert type
#### HED (Holistic edge detection)
Generate HED Data from the base dataset (e.g. Cityscapes)
* apply `datasets/lib_hed/hed.py --image_folder --out_folder` to your RGB images for each split separately or split your data afterwards into train, val and test.

#### EED (Edge Enhancing Diffusion) 
Generate data with the help of the code in folder `Code_EED_generation`

* write a dataloader for the data or adapt your data to the dataloader `CityscapesHEDBlackEdges19classes` in `datasets/expert_datasets.py`
* calculate mean and var for your custom dataset with `calc_dataset_mean_std.py` which uses the `datasets/mean_std.py` script.

## Train shape expert
* Choose the expert config from `configs` or adapt it to your needs.
* Rename it to `config-defaults.yaml`
* Run `main_shape.py`
  
Most important parameters:
  * exp_name
  * dataset type
  * dataset train and val paths
  * normalization parameters

## Evaluation
The expert can be evaluated by running the `main_evaluate_experts.py`
