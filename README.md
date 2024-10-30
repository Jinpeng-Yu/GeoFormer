# GeoFormer: Learning Point Cloud Completion with Tri-Plane Integrated Transformer

![Illustration of the geometry-consistent tri-plane projection in our GeoFormer.](./figures/teaser.png)

This repository contains the PyTorch implementation for:

**GeoFormer: Learning Point Cloud Completion with Tri-Plane Integrated Transformer** (**ACM MM 2024**).  
[Jinpeng Yu](https://scholar.google.com/citations?user=OwkIXhIAAAAJ&hl), Binbin Huang, Yuxuan Zhang, Huaxia Li, Xu Tang, [Shenghua Gao](https://scholar.google.com/citations?hl=zh-CN&user=fe-1v0MAAAAJ)

**[[Paper]](https://arxiv.org/abs/2408.06596)** **[[datasets]](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md)** **[[models]](https://drive.google.com/drive/folders/1Wl8g5N8Utc7IEV9W2nmkRrFr4tyj4w-O?usp=drive_link)**

## Abstract
In this paper, we introduce a GeoFormer that simultaneously enhances the global geometric structure of the points and improves the local details. Specifically, we design a **CCM Feature Enhanced Point Generator** to integrate image features from multi-view consistent canonical coordinate maps (CCMs) and align them with pure point features, thereby enhancing the global geometry feature. Additionally, we employ the **Multi-scale Geometry-aware Upsampler** module to progressively enhance local details. This is achieved through cross attention between the multi-scale features extracted from the partial input and the features derived from previously estimated points.

![Visual comparison with recent methods on ShapeNet55 dataset.](./figures/shapenet55-result.png)

## 🔥News
- **[24.10.30]** Code and pre-trained weights released!
- **[24.10.25]** LaTex Poster for GeoFormer released!

## Pre-trained Models
We provide pre-trained GeoFormer models on PCN and ShapeNet-55/34 benchmarks [here](https://drive.google.com/drive/folders/1Wl8g5N8Utc7IEV9W2nmkRrFr4tyj4w-O?usp=drive_link).

## Usage
### Requirements
- python >= 3.7
- PyTroch >= 1.8.0
- CUDA >= 11.1
- torchvision
- timm
- open3d
- h5py
- opencv-python
- easydict
- transform3d
- tensorboardX
Install PointNet++ and Chamfer Distance.
```
cd pointnet2_ops_lib
python setup.py install

cd metrics/CD/chamfer3D/
python setup.py install
```

### Dataset
Download the [PCN](https://gateway.infinitescript.com/s/ShapeNetCompletion) and [ShapeNet55/34](https://github.com/yuxumin/PoinTr) datasets, and specify the data path in config_*.py (pcn/55).
```
# PCN
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/path/to/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/path/to/ShapeNetCompletion/%s/complete/%s/%s.pcd'

# ShapeNet-55
__C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH     = '/path/to/shapenet_pc/%s'

# Switch to ShapeNet-34 Seen/Unseen
__C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH       = '/path/to/datasets/ShapeNet34(ShapeNet-Unseen21)'
```

### Evaluation
```
# Specify the checkpoint path in config_*.py
__C.CONST.WEIGHTS = "path to your checkpoint"

python main_*.py (pcn/55) --test or
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=13222 --nproc_per_node=1 main_*.py (pcn/55) --test
```

### Training
```
python main_*.py (pcn/55) or
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=13222 --nproc_per_node=8 main_*.py (pcn/55)
```

## Acknowledgements
The repository is based on [SeedFormer](https://github.com/hrzhou2/seedformer), some parts of the code are borrowed from:
- [PoinTr](https://github.com/yuxumin/PoinTr)
- [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet)
- [SVDFormer](https://github.com/czvvd/SVDFormer_PointSea)
- [GRNet](https://github.com/hzxie/GRNet)

We utilize [MeshLab](https://github.com/cnr-isti-vclab/meshlab) to visualize the point cloud completion results.

We thank the authors for their excellent works.

## BibTeX
If you find our work useful in your reasearch, please consider citing:
```
@inproceedings{yu2024geoformer,
  title={GeoFormer: Learning Point Cloud Completion with Tri-Plane Integrated Transformer},
  author={Yu, Jinpeng and Huang, Binbin and Zhang, Yuxuan and Li, Huaxia and Tang, Xu and Gao, Shenghua},
  booktitle={ACM Multimedia 2024}
}
```