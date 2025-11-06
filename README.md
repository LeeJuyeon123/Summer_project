# REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning

This is the repository for the paper 'REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning'


## Abstract

Recent rehearsal-free methods, guided by prompts, generally excel in vision-related continual learning (CL) scenarios with continuously drifting data. To be deployable on real-world devices, these methods must maintain high resource efficiency during training. In this paper, we introduce Resource-Efficient Prompting (REP), which targets improving the resource efficiency of prompt-based rehearsal-free methods. Our key focus is on avoiding catastrophic trade-offs with accuracy while trimming computational and memory costs during prompt learning. We achieve this by exploiting swift prompt selection that enhances input data using a carefully provisioned model, and by developing adaptive token merging (AToM) and layer dropping (ALD) algorithms for the prompt updating stage. AToM and ALD perform selective skipping across the data and model-layer dimensions without compromising task-specific features while learning new tasks. We validate REP’s superior resource efficiency over current state-of-the-art ViT- and CNN-based methods through extensive experiments on three image classification datasets.


## Environment
- Ubuntu 18.04 LTS
- NVIDIA RTX 3090
- Python 3.8.8
```
pytorch==1.12.1 + CUDA 11.3
torchvision==0.13.1
timm==0.6.13
pillow==9.2.0
matplotlib==3.5.3
```
The environments and essential packages can be installed easily by 
```
conda create -n rep python=3.8.8
conda activate rep
conda install -c anaconda cudatoolkit==11.3.1
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
cd rep_code/
pip install -r requirements.txt
```

## Data preparation
If you already have CIFAR-100 or ImageNet-R, pass your dataset path to  `--data-path`.


## Training
To train a model:

```
bash ./experiments/REP_l2p.sh
bash ./experiments/REP_dual.sh
```

## Reference
<a href="https://github.com/JH-LEE-KR/l2p-pytorch">l2p-pytorch</a>

<a href="https://github.com/JH-LEE-KR/dualprompt-pytorch">dualprompt-pytorch</a>

<a href="https://github.com/facebookresearch/ToMe">ToMe</a>