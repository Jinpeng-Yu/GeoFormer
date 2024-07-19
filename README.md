# GeoFormer: Learning Point Cloud Completion with Tri-Plane Integrated Transformer

![Illustration of the geometry-consistent tri-plane projection in our GeoFormer.](./figures/teaser.png)

This repository contains the PyTorch implementation for **GeoFormer: Learning Point Cloud Completion with Tri-Plane Integrated Transformer** (ACM-MM 2024).

[Paper](coming soon) [datasets](coming soon) [models](coming soon) [results](coming soon)

Conventional methods typically predict unseen points directly from 3D point cloud coordinates or use self-projected multi-view depth maps to ease this task. However, these gray-scale depth maps cannot reach multi-view consistency, consequently restricting the performance. In this paper, we introduce a GeoFormer that simultaneously enhances the global geometric structure of the points and improves the local details. Specifically, we design a CCM Feature Enhanced Point Generator to integrate image features from multi-view consistent canonical coordinate maps (CCMs) and align them with pure point features, thereby enhancing the global geometry feature. Additionally, we employ the Multi-scale Geometry-aware Upsampler module to progressively enhance local details. This is achieved through cross attention between the multi-scale features extracted from the partial input and the features derived from previously estimated points. Extensive experiments on the PCN, ShapeNet-55/34, and KITTI benchmarks demonstrate that our GeoFormer outperforms recent methods, achieving the state-of-the-art performance.

![Visual comparison with recent methods on ShapeNet55 dataset.](./figures/shapenet55-result.png)

<!-- ## News -->

<!-- ## BibTeX
If you find our work useful in your reasearch, please cite:
```

``` -->