# Soft-Masked Diffusion Language Models & Spherical Soft-Masking

This repository hosts the official implementation of two papers on Soft-Masked Diffusion Language Models (MDLMs):

1. **Soft-Masked Diffusion Language Models (ICLR 2026)** [[Paper]](https://openreview.net/forum?id=Gba02UMvrG)
2. **Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models (COLM 2026)**

<div align="center">
  <img src='./assets/architecture.png' width="90%"/>
</div>

---

## 💡 Overview

### 1. Soft-Masking (ICLR 2026)
Soft-masking (SM) replaces the standard binary "keep mask or replace" decision in MDLMs with a continuous, confidence-weighted blend of the mask token embedding $\mathbf{m}$ and a weighted superposition of the top-$k$ predicted token embeddings $\mathbf{v}_i$:
$$\tilde{\mathbf{x}}_{t-1} = (1 - \lambda) \cdot \mathbf{m} + \lambda \sum_{i \in \text{top-}k} \pi_i \cdot \mathbf{v}_i$$
where $\lambda \in [0, 1)$ is derived from the model's predictive entropy. This approach accelerates convergence during fine-tuning and continued pre-training.

### 2. Spherical Soft-Masking (COLM 2026)
The follow-up paper, *Lost in Interpolation*, identifies a **norm-collapse failure** in the original linear interpolation (LERP) scheme. Because token embeddings in modern language models exhibit a hyperspherical structure (concentrating on a shell in $\mathbb{R}^D$ where $\mathbf{m}$ and $\mathbf{v}_i$ subtend an angle of $\approx 73^\circ$), straight-line interpolation (LERP) cuts through the hyperspherical shell, systematically shrinking the embedding norm by 20% to 40%. This introduces out-of-distribution inputs that degrade training.

To solve this, we propose **Spherical Soft-Masking (SLERP-SM)**:
* **Weighted Fréchet (Karcher) Mean**: Aggregates the top-$k$ normalized predictions onto the unit sphere $S^{D-1}$ via iterative Karcher flow.
* **Geodesic Interpolation (SLERP)**: Interpolates between the normalized mask embedding and the Fréchet mean along the shared great-circle arc:
$$\mathbf{s} = \frac{\sin((1-\lambda)\Omega)}{\sin\Omega}\hat{\mathbf{m}} + \frac{\sin(\lambda\Omega)}{\sin\Omega}\mathbf{\mu}^*$$
* **Norm Rescaling**: Scales the unit direction vector $\mathbf{s}$ back to the original mask token norm ($r_m = \|\mathbf{m}\|$), ensuring backbone compatibility.

---

## 📊 Key Results

* **Fixed $\lambda = 0.3$ Stable Training**: Linear interpolation (LERP) causes monotonic perplexity degradation. Spherical interpolation (SLERP-SM) remains stable, opening a **3.9 PPL gap** by the end of training on OpenWebText.
* **Generation Quality (MAUVE)**: Under a fixed mixing weight, SLERP-SM yields dramatically higher-quality generations, achieving a **MAUVE score of 0.258** compared to **0.055** for LERP.
* **Learned $\lambda$ Improvement**: Under a learned feedback schedule, SLERP-SM achieves lower NLL and perplexity compared to both LERP and the vanilla MDLM baseline.

---

## 📰 Updates
* **[May 2, 2026]** Checkpoints for language modeling (OWT) released.
* **[Mar 2, 2026]** Initial release.

## Reproducing Results
Please refer to the [Coding](./coding/) and [Language](./language/) folders for experiments with Dream and MDLM, respectively. 

## Checkpoints Language Modeling
Soft-masking checkpoints for language modeling (OpenWebText) can be found in this [Box folder](https://ibm.box.com/v/soft-masked-dlm-checkpoints). 

## Acknowledgement
The coding part was built on top of [Dream-7B](https://github.com/DreamLM/Dream) and the language modeling part based on [Duo](https://github.com/s-sahoo/duo) and ReMDM [ReMDM](https://github.com/kuleshov-group/remdm).

## Citation 📚
If you use the work released here for your research, please consider citing our papers:

```bibtex
@inproceedings{
hersche_softmasking_2026,
title={Soft-Masked Diffusion Language Models},
author={Hersche, Michael and Moor-Smith, Samuel and Hofmann, Thomas and Rahimi, Abbas},
booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
year={2026},
url={https://openreview.net/forum?id=Gba02UMvrG}
}

@article{lost_in_interpolation_2026,
title={Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models},
author={Hersche, Michael and Moor-Smith, Samuel and Hofmann, Thomas and Rahimi, Abbas},
journal={arXiv preprint},
year={2026}
}
```