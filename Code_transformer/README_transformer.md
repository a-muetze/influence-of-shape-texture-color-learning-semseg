# Transformer Experiments

# Cityscapes
Install the following MMSegmentation and MMCV version:
```
mmcv==2.2.0
mmengine==0.10.4
-e git+https://github.com/open-mmlab/mmsegmentation.git@b040e147adfa027bbc071b624bedf0ae84dfc922#egg=mmsegmentation 
```

Replace and add all files in this folder to apply the changes we made to MMSegmentation

## Train a transformer on Cityscapes Anisodiff HS data:
```bash
python tools/train.py zzz_myscripts/segformer_mit-b1_1xb7-170k_cityscapesAnisodiffHS-512x512.py --work-dir segformer_mit-b1_1xb7-170k_cityscapesAnisodiffHS-512x512
```
## Eval a transformer on Cityscapes
```bash
python tools/train.py zzz_myscripts/segformer_mit-b1_1xb7-170k_cityscapesAnisodiffHS-512x512.py --work-dir segformer_mit-b1_1xb7-170k_cityscapesAnisodiffHS-512x512
```
