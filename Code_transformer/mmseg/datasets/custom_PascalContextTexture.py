#Custom dataset based on Cityscapes dataset code of OpenMMLab
# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CustomPascalContextTextureDataset(BaseSegDataset):
    """CustomPascalContext dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to .png' for CustomPascalContext dataset. 
    """
    METAINFO = dict(
        classes=('aeroplane',  
                'bicycle',    
                'bird',       
                'boat',       
                'bottle',     
                'bus',        
                'car',        
                'cat',        
                'chair',      
                'cow',        
                'diningtabel',
                'dog',        
                'horse',      
                'motorbike',  
                'person',     
                'pottedplant',
                'sheep',      
                'sofa',       
                'train',      
                'tvmonitor',  
                'sky',        
                'grass',      
                'ground',     
                'road',       
                'building',   
                'tree',       
                'water',      
                'mountain',   
                'wall',       
                'floor',      
                'track',      
                'keyboard',   
                'ceiling'),
        palette=[   [128,  0,  0],
                    [  0,128,  0],
                    [128,128,  0],
                    [  0,  0,128],
                    [128,  0,128],
                    [  0,128,128],
                    [128,128,128],
                    [ 64,  0,  0],
                    [192,  0,  0],
                    [ 64,128,  0],
                    [192,128,  0],
                    [ 64,  0,128],
                    [192,  0,128],
                    [ 64,128,128],
                    [192,128,128],
                    [  0, 64,  0],
                    [128, 64,  0],
                    [  0,192,  0],
                    [128,192,  0],
                    [  0, 64,128],
                    [  0,192, 64],
                    [  0,  0,192],
                    [128,  0,192],
                    [ 64,128,192],
                    [192, 64,  0],
                    [ 64,192,  0],
                    [192, 64, 64],
                    [ 64,  0, 64],
                    [128, 64,128],
                    [128,  0, 64],
                    [128, 64,192],
                    [  0,128,192],
                    [192,192,  0]
                ]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_train_id.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
