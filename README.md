# Dissecting Image Crops

This is the official repository for B. Van Hoorick and C. Vondrick, "Dissecting Image Crops," *arXiv preprint arXiv:2011.11831*, 2020.
In short, we investigate what traces are left behind by visual cropping.

[Link to paper on ArXiv](https://arxiv.org/pdf/2011.11831.pdf)

![](NewArch_v5.png)

## Basic Usage Instructions

Step 1: Populate `data/train`, `data/val`, and `data/test` with high-resolution image files; a constant aspect ratio is strongly preferred. 

Step 2: Investigate the command line flags in `train.py`, and run `python train.py` with the desired arguments. This will instantiate a new training run with PyTorch checkpoint files in `checkpoints/`, and TensorBoard log files in `logs/`.

Step 3: Run `python test.py --model_path /path/to/above/checkpoint/folder` with relevant arguments to run the model on the test set.

## Dataset

In our project, we scraped [Flickr](https://www.flickr.com/explore) based on [this script](https://github.com/antiboredom/flickr-scrape) by Sam Lavigne, using each line in `google-10000-english-no-swears.txt` (see [this repository](https://github.com/first20hours/google-10000-english) for more info) as search queries. We filtered the photos by an aspect ratio of 1.5, which is the most common value, resulting in a dataset of around 700,000 images.

## Known Issues

There is a stubborn memory leak that builds up as you train over many epochs. I have tried many things but do not know how to prevent it.


## BibTeX Citation

    @article{van2020dissecting,
        title={Dissecting Image Crops},
        author={Van Hoorick, Basile and Vondrick, Carl},
        journal={arXiv preprint arXiv:2011.11831},
        year={2020}
    }
