# SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Expression Segmentation

<div align="center">
  <a href="https://arxiv.org/pdf/2603.18086"><img src="https://img.shields.io/badge/arXiv-2503.12345-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/wayneicloud/SSP-SAM"><img src="https://img.shields.io/badge/HuggingFace-Checkpoint-yellow?style=flat-square" alt="HF Checkpoint"></a>
  <a href="https://huggingface.co/wayneicloud/SSP-SAM"><img src="https://img.shields.io/badge/HuggingFace-Dataset-orange?style=flat-square" alt="HF Dataset"></a>
  <img src="https://img.shields.io/badge/License-Apache--2.0-green?style=flat-square" alt="License">
</div>

<div align="center">
    <a href='https://scholar.google.com/citations?user=D-27eLIAAAAJ&hl=zh-CN' target='_blank'>Wei Tang</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=SVQYcYcAAAAJ' target='_blank'>Xuejing Liu</a><sup>&#x2709,2</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN' target='_blank'>Yanpeng Sun</a><sup>3</sup>&emsp;
    <a href='https://imag-njust.net/zechaoli/' target='_blank'>Zechao Li</a><sup>&#x2709,1</sup>
</div>

<div align="center">
    <sup>1</sup>Nanjing University of Science and Technology;&emsp;
    <sup>2</sup>Institute of Computing Technology, Chinese Academy of Sciences;&emsp;
    <sup>3</sup>NExT++ Lab, National University of Singapore
    <br>
    <sup>&#x2709</sup> Corresponding Authors
</div>

---

## Overview

This repository provides the codebase of **SSP-SAM**, a referring expression segmentation framework built on top of SAM with semantic-spatial prompts.

Current repo status:
- Training/testing/data processing scripts are available.
- Multiple dataset configs are provided under `configs/`.

## 💥 News

- **17 Mar, 2026**: Open-source codebase has been organized and released.
- **4 Dec, 2025**: SSP-SAM paper accepted by IEEE TCSVT.

## 📌 ToDo

- [X] Release final model checkpoints on Hugging Face
- [X] Release processed training/evaluation metadata
- [X] Release arXiv version

## 🔗 Model Zoo & Links

- Paper: [`SSP-SAM`](https://arxiv.org/pdf/2603.18086)
- <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" width="16"/> Hugging Face Checkpoints/datasets: `https://huggingface.co/wayneicloud/SSP-SAM`

## 📁 Project Structure

```text
.
├── configs/                 # training/evaluation configs
├── data_seg/                # data preprocessing scripts and generated anns/masks
├── datasets/                # dataloader and transforms
├── models/                  # SSP_SAM model definitions
├── segment-anything/        # modified SAM dependency (editable install)
├── train.py                 # training entry
├── test.py                  # evaluation entry
├── submit_train.sh          # train launcher (with examples)
└── submit_test.sh           # test launcher (with examples)
```

## ⚙️ Environment Setup

Recommended: conda environment on macOS/Linux.

```bash
conda create -n ssp_sam python=3.10 -y
conda activate ssp_sam
pip install --upgrade pip

# 1) install PyTorch (CUDA example: cu121)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2) install modified segment-anything first
cd segment-anything
pip install -e .
cd ..

# 3) install remaining dependencies
pip install -r requirements.txt
```

> Note: the `segment-anything` code in this repository has been modified based on the original SAM implementation.  
> Please install the local `segment-anything` in editable mode (`pip install -e .`) as shown above.

## 🧩 Data Preparation

Please check:
- `data_seg/README.md`
- `data_seg/run.sh`

You have two options:

1. **Use our provided annotations + generate masks locally (recommended)**  
   - <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" width="16"/> Download `data_seg/anns/*.json` and other prepared `data_seg` files from Hugging Face:  
     `https://huggingface.co/wayneicloud/SSP-SAM`
   - You can directly use our `data_seg/anns/*.json`.
   - `masks` should be generated on your side by running:
     ```bash
     bash data_seg/run.sh
     ```

2. **Regenerate annotations/masks by yourself**  
   See the collapsible section below.

<details>
<summary>Generate Annotations/Masks by Yourself (click to expand)</summary>

References:
- `data_seg/README.md`
- `data_seg/run.sh`
- `legacy_data_prep_simrec.md` (legacy reference for raw data preparation and sources)

Required raw annotation folders/files for generation include (examples):
- `data_seg/refcoco/`
- `data_seg/refcoco+/`
- `data_seg/refcocog/`
- `data_seg/refclef/`

Each folder should contain raw files such as `instances.json` and `refs(...).p`.

Minimal expected layout (example):

```text
data_seg/
├── refcoco/
│   ├── instances.json
│   ├── refs(unc).p
│   └── refs(google).p
├── refcoco+/
│   ├── instances.json
│   └── refs(unc).p
├── refcocog/
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(umd).p
└── refclef/
    ├── instances.json
    ├── refs(unc).p
    └── refs(berkeley).p
```

Example preprocessing command:

```bash
python ./data_seg/data_process.py \
  --data_root ./data_seg \
  --output_dir ./data_seg \
  --dataset refcoco \
  --split unc \
  --generate_mask
```

</details>

Detailed dataset path/config settings are defined in the corresponding preprocessing scripts/config files in `data_seg/`.  
Please modify them according to your local environment before running.
Also check dataset/image path settings in:
- `datasets/dataset.py`

> Important: in `datasets/dataset.py`, class `VGDataset`, you should update local paths for images/annotations/masks according to your machine.

Example local data organization:

```text
your_project_root/
├── data/                                        # set --data_root to this folder
│   ├── coco/
│   │   └── train2014/                           # COCO images (unc/unc+/gref/gref_umd/grefcoco)
│   ├── referit/
│   │   └── images/                              # ReferIt images
│   ├── VG/                                      # Visual Genome images (merge pretrain path)
│   └── vg/                                      # Visual Genome images (phrase_cut path, if used)
└── data_seg/                                    # same level as data/
    ├── anns/
    │   ├── refcoco.json
    │   ├── refcoco+.json
    │   ├── refcocog_umd.json
    │   ├── refclef.json
    │   └── grefcoco.json
    └── masks/
        ├── refcoco/
        ├── refcoco+/
        ├── refcocog_umd/
        ├── refclef/
        └── grefcoco/
```

For training/testing, use:
- `data_seg/anns/*.json` (provided)
- `data_seg/masks/*` (generated locally via `bash data_seg/run.sh`)

### Required Images and Raw Data Sources

For training/evaluation, you need the corresponding image files locally (COCO/Flickr/ReferIt/VG depending on dataset split and config).  
Common sources:
- RefCOCO / RefCOCO+ / RefCOCOg / RefClef annotations: http://bvisionweb1.cs.unc.edu/licheng/referit/data/
- MS COCO 2014 images: https://cocodataset.org/
- Flickr30k images: http://shannon.cs.illinois.edu/DenotationGraph/
- ReferItGame images: due to original dataset restrictions, please download by yourself from the official/authorized source.
- Visual Genome images: https://visualgenome.org/

## 🚀 Training

Default training launcher:

```bash
bash submit_train.sh
```

`submit_train.sh` already includes commented examples for multiple datasets, e.g.:
- `refcoco`
- `refcoco+`
- `refcocog_umd`
- `referit`
- `grefcoco`

You can also run directly:

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/SSP_SAM_CLIP_B_FT_unc.py \
  --clip_pretrained pretrained_checkpoints/CS/CS-ViT-B-16.pt
```

### Resume Modes

`train.py` supports two resume modes:
- `--resume <ckpt>`: use this for interrupted training and continue from the previous checkpoint (断点续训).
- `--resume_from_pretrain <ckpt>`: use this for loading pretrained weights before fine-tuning/training.

## 📊 Evaluation

Default testing launcher:

```bash
bash submit_test.sh
```

Example direct command:

```bash
torchrun --nproc_per_node=1 --master_port=29590 test.py \
  --config configs/SSP_SAM_CLIP_L_FT_unc.py \
  --test_split testB \
  --clip_pretrained pretrained_checkpoints/CS/CS-ViT-L-14-336px.pt \
  --checkpoint output/your_save_folder/checkpoint_best_miou.pth
```

## 📝 Notes

- COCO image path in visualization prioritizes `data/coco/train2014`.
- Current mask prediction/evaluation path uses `512x512` mask space.
- Config files in `configs/` are set with:
  - `output_dir='outputs/your_save_folder'`
  - `batch_size=8`
  - `freeze_epochs=20`

## 🌈 Acknowledgements

This repository benefits from ideas and/or codebases of the following projects:

- SimREC: https://github.com/luogen1996/SimREC
- gRefCOCO: https://github.com/henghuiding/gRefCOCO
- TransVG: https://github.com/djiajunustc/TransVG
- Segment Anything (SAM): https://github.com/facebookresearch/segment-anything

Thanks to the authors for their valuable open-source contributions.

## 📚 Citation

If you find this repository useful, please cite our SSP-SAM paper.

```bibtex
@article{ssp_sam_tcsvt,
  title={SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Expression Segmentation},
  author={Tang, Wei and Liu, Xuejing and Sun, Yanpeng and Li, Zechao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```
