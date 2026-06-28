# AnatoReCT: Anatomy-Aware Conditional Diffusion for Low-Dose CT Reconstruction

## 💡Primary contributions
To address severe quantum noise, structural degradation, and inefficient sampling in low-dose CT (LDCT) reconstruction, we propose AnatoReCT, a lightweight anatomy-aware conditional diffusion framework that explicitly exploits anatomical priors to restore diagnostically reliable normal-dose CT (NDCT) images.

1. Anatomy-Prior Guided Diffusion. We reformulate LDCT reconstruction as a conditional diffusion problem with a mean-preserving degradation process, preserving anatomical structures while reducing sampling complexity.

2. Dual-Branch Anatomy-Aware Feature Fusion. A frequency-decoupled encoder learns complementary low-frequency structural priors and high-frequency detail representations for tissue-adaptive reconstruction.

3. Temporal Gated Fusion. Global anatomical priors and local detail features are dynamically injected across denoising stages to improve structural consistency and fine-detail recovery.

4. State-of-the-Art Performance. AnatoReCT achieves 46.30/37.68 dB PSNR and 0.9794/0.8677 SSIM on the Mayo 2016 and Mayo 2020 datasets, outperforming existing methods in image quality and clinical evaluation.
   
## 🧗Proposed method

![frame](./imgs/frame1.png)

## Table of Contents
- [Requirements](#requirements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)

## 📝Requirements
To install requirements:
```bash
pip install -r requirements.txt
```
