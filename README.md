# Semantic Segmentation for Extracting Geologic Features from Historic Topographic Maps

## Repository Overview

This repository contains the full implementation of a semantic segmentation pipeline for extracting geologic features from historic topographic maps using U-Net and Mask_RCNN.

## Project Structure

```
Semantic-Segmentation-for-Feature-Extraction/
├── main.py               # Entry point: ties together data loading, training, and evaluation
├── loading.py            # Data loading and preprocessing functions
├── trainer.py            # Model architecture, training, and performance tracking
├── evaluate.py           # Model evaluation and visualization
├── utils.py              # Helper functions for plotting, metrics, etc.
├── README.md             # Project documentation
└── models/               # Saved model weights and checkpoints
```


## Objectives

The objective of this project was to extract specific geologic features from historic topographic USGS maps. The goal is to replace time-intensive manual digitization (which can take 1–2 weeks per map) with deep learning techniques that can perform the same task in minutes or hours. Traditional feature extraction methods rely on template matching, which is inadequate for large-scale, noisy datasets such as scanned historical maps. These maps present challenges such as:

* Variability in symbology
* Overprinting with contours, labels, and text
* Scanning artifacts (folds, rotations, blurs)

The objective was to build a model capable of automatically identifying and classifying relevant map features with minimal human intervention, allowing for efficient vectorization and downstream geospatial analysis.

## Model Architecture

This project implements and compares two deep learning models for semantic segmentation: **U-Net** and **Mask R-CNN**. Each model offers distinct advantages depending on the nature of the features being extracted and the characteristics of the input maps.

### U-Net

U-Net is a fully convolutional neural network with an encoder–decoder architecture that has demonstrated remarkable effectiveness in modeling complex spatial patterns. Given the intricate nature of topographic maps and the need for robust pixel-level predictions, U-Net was chosen as a primary architecture. It is particularly effective when working with a limited number of labeled training images and can generalize well to unseen data.

Key strengths of the U-Net approach:

* Well-suited for dense, pixel-level classification
* Captures fine-grained details of geologic features
* Efficient training with relatively few annotated examples

We employed a transfer learning strategy by using a **ResNet34** backbone pre-trained on ImageNet, and experimented with various hyperparameters such as number of layers, kernel sizes, and loss functions (e.g., Focal Loss, Dice Loss, Jaccard Loss).

### Mask R-CNN

Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI). This model is ideal for detecting and segmenting **object-based features** with well-defined boundaries, such as specific geological symbols or discrete features embedded in the map.

Key strengths of the Mask R-CNN approach:

* Excellent at instance segmentation and object detection
* Works well when features of interest are sparse but distinct
* Offers bounding boxes in addition to segmentation masks

The Mask R-CNN implementation provides a complementary approach to U-Net, especially useful when object-level detection is required alongside semantic segmentation.

Both models were implemented from scratch and trained using customized pipelines for data loading, preprocessing, augmentation, and evaluation. Results from each model were compared across multiple metrics including **Intersection over Union (IoU)** and **visual inspection** to assess real-world applicability.

## Network Configuration & Training

* **Loss Functions**: Used categorical focal loss to handle class imbalance and force attention on harder samples. Dice loss and IOU-based losses were also explored.
* **Backbone**: ResNet34 with ImageNet pretrained weights
* **Augmentation**: Applied flipping, scaling, and normalization techniques to improve generalization.
* **Parameter Search**: Systematic experiments were conducted to tune:

  * Number of layers
  * Number of filters per layer
  * Kernel size

Model performance was evaluated using Intersection-over-Union (IoU), Frequency Weighted IoU, accuracy, F1 score, and confusion matrices.

## Results

After training and evaluation:

* Visual comparisons between ground-truth masks and predicted outputs were shared with the client.
* Quantitative metrics confirmed the model's strong performance on unseen data.
* The model performed well on major feature classes, with some confusion in classes underrepresented in training data.

Suggestions were made to include more annotated examples for such classes to improve model precision.

## Repository

GitHub Repository: [https://github.com/hhaeri/Semantic-Segmentation-for-Feature-Extraction](https://github.com/hhaeri/Semantic-Segmentation-for-Feature-Extraction)

Modular refactoring in progress. Current structure includes:

* `main.py`: main execution script
* `loading.py`: data generators and preprocessing
* `trainer.py`: model definition and training loop
* `evaluate.py`: performance evaluation and visualization
* `utils.py`: plotting, augmentation, and helper functions

---

**Author**: Hanieh Haeri
**Created**: January 20, 2023

---
