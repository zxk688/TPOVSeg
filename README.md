# TPOV-Seg
This is a repository for releasing a PyTorch implementation of our work [TPOV-Seg: Textually Enhanced Prompt Tuning of Vision-Language Models for Open-Vocabulary Remote Sensing Semantic Segmentation](https://ieeexplore.ieee.org/document/11215798) accepted for publication in IEEE TGRS.


## Installation


```bash
# 1. Create and activate a conda environment (example with Python 3.10)
conda create -n tpovseg python=3.8 -y
conda activate tpovseg

# 2. Clone the repository and navigate to it
git clone https://github.com/zxk688/TPOVSeg.git
cd TPOVseg

# 3. Install Python dependencies
pip install --upgrade pip
Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Download Datasets
Download the benchmark datasets from public sources, for example:

- Flair: Refer to the [FLAIR](https://ignf.github.io/FLAIR/) 
- GID15: Refer to the [GID-15](https://captain-whu.github.io/GID15/) 
- LOVEDA: Refer to the [Loveda](https://github.com/Junjue-Wang/LoveDA)
- ISPRS Potsdam: Refer to the [Potsdam](https://opendatalab.com/OpenDataLab/ISPRS_Potsdam/)
- ISPRS Vaihingen: Refer to the [Vaihingen](https://opendatalab.com/OpenDataLab/ISPRS_Vaihingen/)
- WHU building: Refer to the [WHU building](https://gpcv.whu.edu.cn/data/building_dataset.html)


### 2. Organize Data Structure
Place all datasets under the `data/` folder at the root of the repository. The recommended structure is:

```
datasets
├──flair
│   ├── annotations
│   │   ├── test
│   │   ├── train
│   │   └── val
│   │    
│   └── images
│       ├── test
│       ├── train
│       └── val
│    
├──gld15
│   ├── annotations
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── images
│       ├── test
│       ├── train
│       └── val
│    
├──LoveDA Urban
│    
├──LoveDA rural
....

```


---

## Training and Testing Examples

Run the following commands from the repository root:

```bash
# train on datasets
python train_net.py \
  --config ./configs/vitb_train.yaml \
  --num-gpus 1 \
  --dist-url "auto" \
  OUTPUT_DIR output/

# val on datasets
python train_net.py \
  --config configs/vitb_384_test.yaml \
  --num-gpus 1 \
  --dist-url "auto" \
  --eval-only \
  OUTPUT_DIR output/eval \
  MODEL.WEIGHTS ./output/model_final.pth
```

---

## Citation

If you find this repository useful for your research, please cite our paper:
```
@article{Zhang2025,
  title = {TPOV-Seg: Textually Enhanced Prompt Tuning of Vision-Language Models for Open-Vocabulary Remote Sensing Semantic Segmentation},
  ISSN = {1558-0644},
  url = {http://dx.doi.org/10.1109/TGRS.2025.3624767},
  DOI = {10.1109/tgrs.2025.3624767},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Zhang,  Xiaokang and Zhou,  Chufeng and Huang,  Jianzhong and Zhang,  Lefei},
  year = {2025},
  pages = {1–1}
}
```

## Acknowledgements 
This codebase is heavily borrowed from [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg). We sincerely thank the authors for their valuable efforts.


