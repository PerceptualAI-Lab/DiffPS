<div align="center">

# [ICCV 2025 Highlight] DiffPS: Leveraging Prior Knowledge of Diffusion Model for Person Search

<div>
    <a href='https://sites.google.com/view/pai-lab/home?authuser=0' target='_blank'>Giyeol Kim</a><sup>1*</sup> &nbsp;
    <a href='https://ncia.snu.ac.kr/' target='_blank'>Sooyoung Yang</a><sup>2*</sup> &nbsp;
    <a href='https://cmlab.cau.ac.kr/our-team/professor' target='_blank'>Jihyong Oh</a><sup>1</sup> &nbsp;
    <a href='https://ncia.snu.ac.kr/general-5-1' target='_blank'>Myungjoo Kang</a><sup>2,3</sup> &nbsp;
    <a href='https://sites.google.com/view/pai-lab/members/faculty?authuser=0' target='_blank'>Chanho Eom</a><sup>† 1</sup>
</div>
<div>
    <sup>1</sup>GSAIM, Chung-Ang University &nbsp;|&nbsp; <sup>2</sup>IPAI, Seoul National University
</div>
<div>
    <sup>3</sup>Department of Mathematical Sciences and RIMS, Seoul National University
</div>
<div>
    <sup>*</sup>Co-first authors (equal contribution) &nbsp;|&nbsp; <sup>†</sup>Corresponding author
</div>

</div>

---

## News

- **July 24, 2025**: **DiffPS is selected as a Highlight Paper** at ICCV 2025.
- **June 24, 2025**: DiffPS is accepted to ICCV 2025.

---


## Installation

### 1. Diffusion feature extraction (required, do this first)

**Environment setup for diffusion features is required.** Follow the installation guide in [generic-diffusion-feature](https://github.com/Darkbblue/generic-diffusion-feature) (NeurIPS'24) to configure your environment. That repository provides the diffusion model setup, dependency versions, and layer interfaces used in this codebase.

### 2. Install dependencies from requirements.txt

```bash
pip install -r requirements.txt
```

### 3. Install pytorch_wavelets

This project uses [pytorch_wavelets](https://pytorch-wavelets.readthedocs.io/en/latest/readme.html) for the wavelet transform. Install from the included submodule or from the official repo:

```bash
# Option A: install from the included pytorch_wavelets folder
cd pytorch_wavelets && pip install . && cd ..

# Option B: install from official repo
# git clone https://github.com/fbcotter/pytorch_wavelets
# cd pytorch_wavelets && pip install . && cd ..
```

---

## Dataset

Place the dataset under the path specified in the config:

- **CUHK-SYSU**: set `DATASET.PATH` in `configs/cuhk_sysu.yaml` (or `configs/_path_cuhk_sysu.yaml`).
- **PRW**: set `DATASET.PATH` in `configs/prw.yaml` (or `configs/_path_prw.yaml`).

Default path in the provided configs is `../dataset/PRW` or `../dataset/CUHK-SYSU`. Adjust to your directory structure.

---

## Training

Run training with the provided script:

```bash
./train.sh
```

Or run directly with a config and options:

```bash
# PRW dataset
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/prw.yaml \
    --opts OUTPUT_DIR "DiffPS" DATASET.BATCH_SIZE 3

# CUHK-SYSU dataset
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/cuhk_sysu.yaml \
    --opts OUTPUT_DIR "DiffPS_CUHK" DATASET.BATCH_SIZE 3
```

You can override any config key via `--opts`, e.g. `OUTPUT_DIR`, `DATASET.BATCH_SIZE`, `DATASET.PATH`.

---

## ⚠️ Note on GPU memory (xformers / cross attention)

**Cross attention map extraction** in the diffusion backbone can consume a large amount of GPU memory depending on the **xformers** (and related) version. If you run into OOM (out-of-memory) errors:

- Try reducing `DATASET.BATCH_SIZE` in the config or `--opts`.
- Check compatibility between your PyTorch, CUDA, and xformers versions; some xformers builds use more memory for attention.
- Consider using a GPU with more VRAM when training with cross-attention features enabled.

**Tip:** Extracting **diffusion features** and **cross attention maps** once and **saving them locally**, then loading from disk during training, is much faster and more memory-efficient than computing them on the fly. We recommend this approach when GPU memory is limited.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2025leveraging,
  title={Leveraging prior knowledge of diffusion model for person search},
  author={Kim, Giyeol and Yang, Sooyoung and Oh, Jihyong and Kang, Myungjoo and Eom, Chanho},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20301--20312},
  year={2025}
}
```

---

## References

- [SEAS](https://github.com/whbdmu/SEAS) — Person search method. This codebase is built upon SEAS.
- [generic-diffusion-feature](https://github.com/Darkbblue/generic-diffusion-feature) — Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features (NeurIPS 2024).
- [pytorch_wavelets](https://pytorch-wavelets.readthedocs.io/en/latest/readme.html) — 2D discrete / dual-tree complex wavelet transforms in PyTorch.
