# Preprocessing for HR-VITON

## 1. Openpose

Build openpose(https://github.com/CMU-Perceptual-Computing-Lab/openpose) on any OS, and run the command below.
```
openpose.bin --image_dir {image_path} --hand --disable_blending --display 0 --write_json {save_path} --num_gpu 1 --num_gpu_start 0
```
Please check the detail arguments on the openpose github. "openpose.bin" could be changed depending on your OS (e.g. Windows -> openpose.exe)
You can get ".json" format files for two clothing-agnostic images; parse and human image.

## 2. Human parse
Check https://github.com/Engineering-Course/CIHP_PGN for human parsing.
I inferenced a parse map on 256x192 resolution, and upsample it to 1024x768.
Then you can see that it has a alias artifact, so I smooth it using "torchgeometry.image.GaussianBlur((15, 15), (3, 3))".
I saved a parse map image using PIL.Image with P mode.
The color of the parse map image in our dataset(VITON-HD) is just for the visualization, it has 0~19 uint values.

## 3. Densepose
Please check the 'detectron2' repository. It doesn't matter if you use UV map or parse map for the reproduction on your custom datasets.
Unfortunately, even I have difficulty reproducing densepose images exactly on the dataset.
Please note that I am currently working on a new dataset using the dense pose image obtained by DumpAction in 'apply_net.py'.
It seems that there is no significant problem with performance of HR-VITON.

## 4. cloth mask
I think you can obtain it using any computer vision methods or neural network model.
The model I used is on "https://github.com/OPHoperHPO/image-background-remove-tool".
I used older version, but I think it will be okay with any version.

## 5. Parse agnostic
It is a bit messy, you can get a parse agnostic image by the code 'get_parse_agnostic.py'.
Please check the code :)
