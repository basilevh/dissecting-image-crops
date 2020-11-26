# Dissecting Image Crops

[Link to paper on ArXiv](https://arxiv.org/pdf/2011.11831.pdf)

![](NewArch_v5.png)

## Minimal Usage Instructions

Step 1: Populate `data/train`, `data/val`, and `data/test` with high-resolution image files.

Step 2: Run `python train.py`.

Step 3: Run `python test.py --model_path /path/to/above/checkpoint/folder`.

## Known Issues

* There is a stubborn memory leak that builds up as you train over many epochs. I have tried many things but have no idea how to prevent it.
