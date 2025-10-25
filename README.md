# On the influence of shape, texture and color for learning semantic segmentation
Official repository of the Paper "On the Influence of Shape, Texture and Color for Learning Semantic Segmentation" accapted at ECAI 2025.

The code includes the cue decomposition procedures, our python configurations and dataset splits.

We masked paths when pointing to the authors' directory. So please adapt them before use. All code and examples are set to Cityscapes as base dataset and its specific folder structure and naming. Changing the dataset includes changes in paths in multiple files and switching the Dataloader.
Path handling as well as the overall usability is work in progress.


## Structure of the repository

* Code with respect to DeepLabV3 with ResNet18 is provided in `Code_CNN`<br>
* Code with respect to SegFormer-B1 is provided in `Code_transformer`
* Code for texture generation is provided in `Code_texture_generation`<br>
* Code for shape generation based on EED is provided in `Code_EED_generation`<br>
* Lists of all used images and the splitting in train, val and test is given in `Dataset_splits`


For the transformer setup we build upon MMSegmentation. To reveal the modifications we made, we provide only the modified files in the file structure identically to MMSegmentation. The used MMSegmentation version can be found in `Python_configurations` 
 
Further details are given in the README_xxx.md in the specific folders.

Remark: Throughout the code we often use anisodiff to refer to S_{EED-xx}.
