# Semantic Segmentation with U-Net for Leukocyte Images

*Please, download the `weights.zip` file in the tag section before running the main.py*

## Claim
From a Diana MATEUS Lab work, lesson 'APSTA' from Centrale Nantes.

## Overview

![Predictions](https://github.com/user-attachments/assets/d0181b2a-d9e8-42c4-b6b5-89e08f50b830)

This project explores the application of semantic segmentation using the U-Net architecture for analyzing leukocyte images. Semantic segmentation is essential in medical imaging for identifying and classifying different regions within an image at the pixel level, aiding in the diagnosis of various hematological diseases.

## U-Net Architecture

### Description

U-Net is a convolutional network designed for fast and precise image segmentation. It is particularly effective in biomedical image segmentation tasks.

### Structure

- **Contracting Path**: Down-samples the image to capture context.
- **Expanding Path**: Up-samples to enable precise localization.

## Application in Leukocyte Segmentation

### Importance

In the context of leukocyte images, semantic segmentation helps in accurately identifying and isolating different types of white blood cells from microscopic images. This process is crucial for diagnosing diseases such as leukemia.

### Challenges

- Complex nature of cell images.
- Variations in staining techniques and imaging conditions.

## Data Augmentation

### Techniques

To enhance the generalizability of the model, data augmentation techniques are employed:

- Brightness and contrast adjustments.
- Flipping and random shearing.

## Deep Learning Techniques

### Models

Deep learning models, including U-Net, utilize convolutional neural networks (CNNs)
