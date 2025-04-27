# CNN-Alzheimers-ImageProcessing
# Alzheimer's MRI Classification Project

## Overview
This project develops a deep learning model to classify brain MRI scans into four stages of Alzheimer's disease:

- Non-Demented
- Very Mild Dementia
- Mild Dementia
- Moderate Dementia

A key feature is the integration of Grad-CAM visualization to highlight regions of the MRI that influenced the classification, providing interpretability to the model.

## Problem Statement
Alzheimer's disease affects over 55 million people worldwide. Diagnosis typically relies on MRI interpretation by specialists, which can be subject to human error. This project aims to create an automated system that can accurately classify Alzheimer's stages while providing visual explanations for its decisions.

## Input and Output

- **Input:** 2D brain MRI scan images (.jpg format)
- **Output:**
  - Classification label (one of four Alzheimer's stages)
  - Grad-CAM heatmap highlighting key activation regions

## Dataset
The project uses the "Augmented Alzheimer MRI Dataset" from Kaggle, containing approximately 6,400 images across the four categories (both original and augmented images).

- **Dataset Link:** https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

Ensure the directory structure looks like for training (change the folder names corresponding to the names below):

```
data/
├── non_demented/
├── very_mild_dementia/
├── mild_dementia/
├── moderate_dementia/
```

## Model Architecture
The model implements a dual-path CNN architecture inspired by this research:

- **Reference Paper:** https://www.nature.com/articles/s41598-024-53733-6
- **Path 1:** Small filters (3×3) capturing fine details through progressive convolutional layers with pooling
- **Path 2:** Larger filters (5×5) capturing broader patterns through parallel convolutional layers with pooling
- **Classification:** Features from both paths are concatenated and processed through fully connected layers

This design was chosen because it is:

- Lightweight, allowing for faster training and inference
- Capable of delivering 90%+ accuracy

I tuned down the parameters for my laptop to ensure it trains efficiently on limited hardware.

## Setup & Installation

### Requirements
- torch>=1.8.0
- torchvision>=0.9.0
- numpy>=1.19.5
- matplotlib>=3.3.4
- scikit-learn>=0.24.1
- Pillow>=8.2.0
- tqdm>=4.59.0
- opencv-python>=4.5.1

## Instructions for Running the Code

### Usage

#### Training
To train the model (update folder names as shown in **Dataset**):
```bash
python train.py
```

#### Prediction
To analyze MRI images:
```bash
python predict.py
```
This automatically processes the 40 images in `data/` (no need to specify an input folder).

To generate Grad-CAM visualizations:
```bash
python predict.py --mode gradcam
```
