# TBD
This repository contains a PyTorch implementation for our paper High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions.

![Untitled](./figures/fig.png)

## Inference

Here are the download links for each model checkpoint:

- Try-on condition generator: [link](https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ/view?usp=sharing)
- Try-on image generator: [link](https://drive.google.com/file/d/1BkSA8UJo-6eOkKcXTFOHK80Esc4vBmVC/view?usp=sharing)
- AlexNet (LPIPS): [link](https://drive.google.com/file/d/1FF3BBSDIA3uavmAiuMH6YFCv09Lt8jUr/view?usp=sharing), we assume that you have downloaded it into `./eval_models/weights/v0.1`.

```python
python3 test_SPADE.py --occlusion --mtviton_checkpoint <condition generator ckpt> --fp16 --gpu_ids 0 --gen_checkpoint <image generator ckpt> --datasetting unpaired --dataroot __ --data_list __
```

## Train try-on condition generator

```python
python3 train.py --gpu_ids 0 --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion --dataroot __ --test_dataroot __ 
```

## Train try-on image generator

```python
python3 train_generator.py --name test -b 4 --j 8 --gpu_ids 0,1 --fp16 --dataroot __ --test_dataroot __ --mtviton_checkpoint <condition generator ckpt path> --occlusion
```
This stage takes approximately 4 days with two RTX 3090 GPUs. Tested environment: PyTorch 1.8.2+cu111.