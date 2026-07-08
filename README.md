# Eye Anterior Segment Spot-like Anomalies Dataset

## Introduction
This repository contains the dataset for the paper: **"Accurate Segmentation of Bulbar Conjunctival Spot-like Anomalies in Eye Anterior Segment Images via a Boundary-Aware Generative Framework"**. 

The dataset is specifically constructed for the automated quantitative analysis and segmentation of spot-like anomalies on the bulbar conjunctiva (caused by lesions, pigmentation, or tissue degeneration) in eye anterior segment images.

## Dataset Details
- **Image Acquisition:** Captured using the MyEyeD-10 shadowless imaging system in collaboration with a hospital in Hangzhou.
- **Preprocessing:** Regions of Interest (ROIs) were extracted using a U-Net++ model (EfficientNetB3 backbone) for sclera segmentation, yielding images with a black background and white sclera.
- **Volume:** The dataset comprises a total of **120** eye anterior segment images.
- **Annotations:** Pixel-level manual annotations of the spot-like anomalies were performed using LabelMe software under the strict guidance of specialized physicians.
- Code Availability
The official PyTorch implementation of the GAB-Seg framework proposed in our paper is currently not publicly available in this repository.

If you are interested in our code or wish to request it for academic research purposes, please feel free to contact the first author:

Zhaocheng Fu: fzhaocheng@foxmail.com

## Citation
If you find this dataset useful for your research, please consider citing our paper:

```bibtex
@article{fu2026accurate,
  title={Accurate Segmentation of Bulbar Conjunctival Spot-like Anomalies in Eye Anterior Segment Images via a Boundary-Aware Generative Framework},
  author={Fu, Zhaocheng and Zhou, Houpan and Luo, Le},
  journal={To be updated},
  year={2026},
  publisher={MDPI}
}
