# Texture Generation Procedure
Install the necessary packages via pip `pip install -r requirements.txt`

To generate a semantic segmentation task based on the CITYSCAPES texture and Voronoi cells follow these three steps:

## 1) Generation of mosaic images
Adapt the upsampling factor per class in the `texture-voronoi-diagrams/polygon_semseg.py` file to upsample texture patches according to the needs of the considered dataset.

Run
```
python3 texture-voronoi-diagrams/polygon_semseg.py --label_id <id>  --dataset_root_path <Cityscapes_path>  --result_path <result_path_for_mosaics> --split <data_split>
```
for a single class or use a bash file to iteratively generate mosaic images for all classes: 

```
#!/bin/bash
data_split="train" 
ids="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"

echo $data_split
for id in $ids; do
    echo $id
    python3 texture-voronoi-diagrams/polygon_semseg.py --dataset_root_path <Cityscapes_path> --label_id $id --split $data_split
done

```


## 2) Generation of contour filled images
```
python3 texture-voronoi-diagrams/polygon_semseg_contourfill.py --label_id <id> --dataset_root_path <Cityscapes_path> --texture_root_path <path_to_mosaics>  --split <'train' | 'val' | 'test'>
```

 
 ## 3) Voronoi diagram

In the script `Voronoi.py`, adapt all parameters in the `"__main__"` function according to your needs. 
For Cityscapes we used:

    cell_number = 100
    dimensionen = 2
    diagramm_amount = 3000
    root_path = "texture-voronoi-diagrams/"
    base_prefix = "upsampled_texture_images"
    split = 'train'
    result_dir = f"Voronoi_{base_prefix}"
    label_list = []
    num_ids=19
    rng = np.random.default_rng(4224)

Then run
`python3 texture-voronoi-diagrams/voronoi.py`
