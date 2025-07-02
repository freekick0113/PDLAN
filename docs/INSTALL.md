## Installation

### Requirements
- Linux
- Python 3.8  
- PyTorch 1.7.0 
- CUDA 11.0 
- NCCL 2+
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [BDD100K api](https://github.com/bdd100k/bdd100k)

### Install PDLAN

a. Create a conda virtual environment and activate it.
```shell
conda create -n pdlan python=3.8 -y
conda activate pdlan
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

c. Install mmcv and mmdetection.

```shell
pip install mmcv-full==1.2.7
pip install mmdet==2.10.0
```

You can also refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md).

Note that mmdetection uses their forked version of pycocotools via the github repo instead of pypi for better compatibility. If you meet issues, you may need to re-install the cocoapi through
```shell
pip uninstall pycocotools
pip install mmpycocotools
```

d. Install mot metrics
```shell
pip install motmetrics
```

e. Install PDLAN
```shell
python setup.py develop
```
