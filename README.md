# Detect Pneumonia AI

An explainable AI research prototype for pneumonia classification from chest X-ray images.

> **Disclaimer:** This project is developed for educational and research purposes.
> It is not a clinically validated diagnostic system and must not be used for medical diagnosis or clinical decision-making.

---

## 1. Overview

This project explores the development of an explainable deep learning system for binary pneumonia classification from chest X-ray (CXR) images.

Given a single chest X-ray image as input, the system aims to:

1. Classify the image as:
   - `NORMAL`
   - `PNEUMONIA`
2. Produce a model score representing the estimated likelihood of the `PNEUMONIA` class.
3. Evaluate model performance using clinically relevant classification metrics.
4. Analyze false-positive and false-negative predictions.
5. Visualize image regions contributing to model predictions using Grad-CAM.

The project is intended to serve as an academic AI/ML learning project and research prototype, with an emphasis on understanding, reproducibility, evaluation, error analysis, and explainability.

---

## 2. Project Objectives

The main objectives are:

- Build a clean and modular deep learning pipeline for chest X-ray classification.
- Establish a reproducible baseline model using transfer learning.
- Prevent or identify potential data leakage during dataset validation.
- Compare different preprocessing, augmentation, loss, and model configurations.
- Evaluate models using more than simple classification accuracy.
- Investigate model errors through false-positive and false-negative analysis.
- Apply Grad-CAM to visualize model attention and investigate whether predictions are based on meaningful image regions.
- Document experimental decisions and findings throughout the project.

---

## 3. Problem Definition

### Input

A single chest X-ray image in JPG format.

### Output

The system produces:

- A binary classification:
  - `NORMAL`
  - `PNEUMONIA`
- A pneumonia model score/probability.
- A final prediction determined using a configurable classification threshold.

For example:

```text
Pneumonia score: 0.87
Threshold: 0.50
Prediction: PNEUMONIA

## 4. End-to-end pipeline
JPG Chest X-ray
        │
        ▼
┌─────────────────────────────┐
│ 01. Dataset Validation      │
│                             │
│ - Train / Val / Test check  │
│ - Patient leakage check     │
│ - Duplicate check           │
│ - Class distribution        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 02. Preprocessing           │
│                             │
│ - Load image                │
│ - Channel handling          │
│ - Resize                    │
│ - Optional ROI              │
│ - Optional CLAHE            │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
     TRAIN         VAL / TEST
        │             │
        ▼             │
┌───────────────┐     │
│ 03. Augment   │     │
│               │     │
│ - Geometric   │     │
│ - Intensity   │     │
│ - Mixup/CutMix│     │
└───────┬───────┘     │
        │             │
        └──────┬──────┘
               ▼
┌─────────────────────────────┐
│ 04. Normalization           │
│                             │
│ ImageNet / Dataset Stats    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 05. Backbone Network        │
│                             │
│ DenseNet121 / ResNet /      │
│ EfficientNet / Swin-T       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 06. Classification Head     │
│                             │
│ Feature → Dropout → Linear  │
│ Output: 1 Logit             │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
    TRAINING       INFERENCE
        │             │
        ▼             ▼
 BCE / Weighted BCE  Sigmoid
 Focal / Asym Loss      │
        │               ▼
        ▼              TTA?
 Backpropagation        │
        │               ▼
        ▼         Calculated Probability
 Model Checkpoint       │
                        ▼
                   Thresholding
                        │
                        ▼
                  Final Prediction
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
      Comprehensive Eval     Error Analysis
              │               FP / FN
              │                   │
              └─────────┬─────────┘
                        ▼
                 Grad-CAM / XAI
                        │
                        ▼
                 Quality Verification

Về sau viết tiếp.
