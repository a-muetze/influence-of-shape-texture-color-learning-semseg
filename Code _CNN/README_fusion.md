# Fusion of two experts

Set `save_softmax` to True and run 
```
python main_infer_and_visualize_results.py
```
for the experts you want to fuse to get the softmax activations for train, val and test. 

For a more stable training calculate the normalization parameters with the help of `calc_dataset_mean_std.py` for the softmax train data.

Adapt parameters in the config `config-defaults_fuse_texture_RGB_anisodiff_RGB.yaml`, e.g. the softmax normalization parameters and rename it to `config-defaults.yaml`

Then run 
```
python main_expert_fusion.py
```
to train the fusion model.

The trained model can be evaluated with `main_evaluate_fusion.py`
