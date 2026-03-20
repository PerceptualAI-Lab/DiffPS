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

<div>
    <a href='https://perceptualai-lab.github.io/DiffPS/'>Project Page</a> &nbsp;|&nbsp;
    <a href='https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Leveraging_Prior_Knowledge_of_Diffusion_Model_for_Person_Search_ICCV_2025_paper.pdf'>Paper (ICCV)</a> &nbsp;|&nbsp;
    <a href='https://arxiv.org/pdf/2510.01841'>arXiv</a>
</div>

</div>

---

## News

- **July 24, 2025**: **DiffPS is selected as a Highlight Paper** at ICCV 2025.
- **June 24, 2025**: DiffPS is accepted to ICCV 2025.

---


## Installation

### 1. Diffusion feature extraction

**Environment setup for diffusion features is required.** Follow the installation guide in [generic-diffusion-feature](https://github.com/Darkbblue/generic-diffusion-feature) (NeurIPS'24) to configure your environment. That repository provides the diffusion model setup, dependency versions, and layer interfaces used in this codebase.

Optional: if you clone it next to this repo as `generic-diffusion-feature/`, `models/diffps.py` will pick up `generic-diffusion-feature/feature` automatically. Otherwise set **`DIFFUSION_FEATURE_PATH`** to the absolute path of that `feature` directory.

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

For downloading and preparing **CUHK-SYSU** and **PRW**, please refer to the dataset instructions in [SeqNet](https://github.com/serend1p1ty/SeqNet/tree/master).

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

## ⚠️ Note on GPU memory (xformers)

There are **known issues with certain xformers versions** that can cause high GPU memory usage when the diffusion backbone computes cross-attention maps (e.g., for DGRPN). If you run into OOM (out-of-memory) errors:

- Try reducing `DATASET.BATCH_SIZE` in the config or `--opts`.
- Check **xformers compatibility** with your PyTorch and CUDA versions; some builds are known to use significantly more memory for attention. Updating or switching xformers version may help.
- Consider using a GPU with more VRAM if the issue persists.

**Tip:** Extracting **diffusion features** and **cross attention maps** once and **saving them locally**, then loading from disk during training, is much faster and more memory-efficient than computing them on the fly. We recommend this approach when GPU memory is limited.

**Note:** We previously experienced an unexpected code loss and have now almost fully recovered the repository. If you have any questions or find something missing, feel free to ask.

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
