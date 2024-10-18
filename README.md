# PixelGaussian: Generalizable 3D Gaussian Reconstruction From Arbitrary Views

<!--paper link and url-->
### [Paper]() | [Prject Page]()

> [Xin Fei](https://scholar.google.com/citations?hl=zh-CN&user=r9rsD_0AAAAJ), [Wenzhao Zheng](https://wzzheng.net/), [Yueqi Duan](https://duanyueqi.github.io/), [Wei Zhan](https://zhanwei.site/), [Masayoshi Tomizuka](https://me.berkeley.edu/people/masayoshi-tomizuka/), [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)

PixelGaussian pioneers to generate **context-aware** Gaussian distributions for feed-foward 3D reconstruction compared with existing uniform pixel-wise paradigm.

![teaser](./assets/teaser.png)

## News
- Training code coming soon.
- Project page coming soon.
- **[2024/10/18]** Paper released on [arXiv]().
- **[2024/10/18]** Demo released.

## Demo
![demo](./assets/demo.gif)

## Overview

![pipeline](./assets/pipeline.png)

Existing generalizable 3D Gaussian splatting methods for 3D reconstruction typically assign a fixed number of Gaussians to each pixel, leading to inefficiency in capturing local geometry and overlap across views. In comparison, we propose a PixelGaussian model consisting of Cascade Gaussian Adapter (CGA) and Iterative Gaussian Refiner (IGR) blocks. In CGA, the initial Gaussians goes through adaptive splitting and pruning operations guided by a keypoint scorer and context-aware hypernets. After CGA, more Gaussians are allocated in regions with rich geometric details, while duplicated and redundant Gaussians across views are removed. Furthermore, to enable such adaptive Gaussians to fully capture local information within images, IGR refines Gaussian representations via deformable attention between image features and
Gaussian queries. With comparable efficiency, our PixelGaussian achieves an average PSNR improvement of around 6 dB in 3D reconstruction from arbitrary views.

![block illustration](./assets/block_illustration.png)
