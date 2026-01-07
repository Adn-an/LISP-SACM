# Lung Infection Severity Prediction with Score-correlated Anatomical CutMix augmentation (LISP-SACM)

## 1\. Overview

This code implements a transformer-based approach for quantifying lung infection severity (Geographic Extent and Lung Opacity) from Chest X-rays. 
The core contribution is the **Score-correlated Anatomical CutMix (SACM)**, a data augmentation strategy that utilizes anatomical lung segmentation to perform semantically informed image mixing.

**Note:** The base implementation of the ViTReg-IP model is accessible at https://github.com/bouthainas/ViTReg-IP.

## 2\. Dependencies

To run this code, the following libraries are required:

  * `torch`
  * `timm` (for the Vision Transformer backbone)
  * `pytorch_lightning`
  * `torchmetrics`
  * `numpy`

## 3\. Model Architecture

The model utilizes the `vit_tiny_patch16_224` architecture pretrained on ImageNet.

  * **Backbone:** Vision Transformer (ViT).
  * **Head:** A two-layer regression head (`Linear(192, 128) -> Linear(128, 2)`) replaces the default classifier to predict pneumonia severity scores.
  * **Objective:** The model minimizes L1 Loss between predicted and ground truth scores.

## 4\. Score-correlated Anatomical CutMix (SACM) Algorithm

The `anatomical_cutmix_multi_zone` method in the code implements the SACM pipeline described in the paper.

### 4.1. Region Selection

The lungs are segmented into six non-overlapping sub-regions (Upper, Middle, Lower for both Left and Right lungs). The algorithm selects $N$ masks (where $N \in \{1...6\}$) from a source image $A$ to paste onto a destination image $B$.

### 4.2. Center-Coinciding Alignment

Unlike standard CutMix, SACM respects anatomy. The `calculate_centers_of_mass` function computes the centroid of the anatomical regions.

  * The algorithm calculates the shift vector $\Delta_i$ required to align the center of the source region mask $M^A_i$ with the center of the destination region mask $M^B_i$.
  * The source region is shifted by $\Delta_i$ before pasting, ensuring the anatomical structure (e.g., lower left lung) is placed correctly on the destination image.

### 4.3. Label Mixing

The new label score $S_C$ for the augmented image is calculated based on the number of regions transferred $N$:

$$S_{C}=\frac{N}{6}S_{A}+(1-\frac{N}{6})S_{B}$$

Where $S_A$ is the score of the source image and $S_B$ is the score of the destination image.

## 5. Citation

If you use this code or the results in your research, please cite the original article:

```
@inproceedings{MustaficSACM,
  author = {Mustafic, Adnan and Dornaika, Fadi and Hammoudi, Karim},
  booktitle = {Proceedings of 2025 International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD 2025) - Lecture Notes in Electrical Engineering},
  publisher = {Springer},
  title = {{Lung Infection Severity Prediction with Parallel Transformers and Score-correlated Anatomical CutMix Augmentation}},
  year = {2025}
}
```
