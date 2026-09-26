# ⚡ Real-Time Face Mask Detection System

> A production-grade, real-time computer vision application engineered from scratch using **Python** 🐍, **TensorFlow / Keras** 🧠, **OpenCV** 👁️, and **MobileNetV2** 🚀. 

---

## 📌 Project Essence & Portfolio Highlights
Designed specifically for campus placement interviews and industry showcases, this project demonstrates advanced, end-to-end deep learning engineering workflows:
* **🔥 Transfer Learning & Fine-Tuning:** Leveraging pre-trained ImageNet feature backbones with selective layer unfreezing.
* **⚡ Real-Time Edge Inference:** Low-latency webcam stream processing with dynamic region-of-interest (ROI) bounding boxes.
* **🛡️ Optimized Data Pipelines:** High-performance caching, batching, and precise pixel scaling (`[-1, 1]`).

---

## 🗂️ Table of Contents
1. [Dataset Description](#-dataset-description)
2. [Theoretical Foundation & Architecture](#-theoretical-foundation--architecture)
3. [Workflow & Pipeline](#-workflow--pipeline)
4. [Training & Fine-Tuning Strategy](#-training--fine-tuning-strategy)
5. [Performance & Accuracy Metrics](#-performance--accuracy-metrics)
6. [Project Directory Structure](#-project-directory-structure)
7. [Installation & Execution Guide](#-installation--execution-guide)

---

## 📁 1. Dataset Description
* **Dataset Source:** *Face Mask Detection ~12K Images Dataset* (Ashish Jangra).
* **Classes:** 
  * `With Mask` (Index 0)
  * `Without Mask` (Index 1)
* **Structure:** Partitioned into dedicated `Train` and `Validation` split directories to ensure rigorous evaluation and prevent data leakage.

---

## 🧠 2. Theoretical Foundation & Architecture

### Why MobileNetV2 over Custom CNNs?
* **The Limitations of Scratch CNNs:** Custom Convolutional Neural Networks trained from scratch often overfit on small datasets and struggle with real-world noise like indoor shadows, varied lighting, and facial hair (e.g., mistaking a beard for a dark mask).
* **Transfer Learning Advantage:** MobileNetV2 is pre-trained on millions of ImageNet images. It natively understands complex human facial geometries, skin textures, and hierarchical edges.
* **Inverted Residuals & Linear Bottlenecks:** MobileNetV2 uses lightweight depthwise separable convolutions, making it exceptionally fast and optimized for real-time edge streaming without hardware lag.

---

## 🔄 3. Workflow & Pipeline
1. **Data Ingestion:** Images are loaded dynamically using `image_dataset_from_directory` with batching and categorical encoding.
2. **Preprocessing:** Input dimensions are standardized to `(224, 224, 3)` with MobileNetV2 specific pixel scaling via `preprocess_input` (`[-1, 1]` range).
3. **Face Localization:** Real-time face coordinates `(x, y, w, h)` are detected frame-by-frame using OpenCV's classical Haar Cascade classifier (`haarcascade_frontalface_default.xml`).
4. **Inference & UI Rendering:** Cropped face regions are passed to the fine-tuned model for multi-class probability estimation, rendering dynamic bounding boxes (Green for Mask, Red for No Mask) with live confidence overlays.

---

## ⚙️ 4. Training & Fine-Tuning Strategy
The model was trained in two robust phases:
* **Phase 1 (Classifier Head Training):** 
  * The MobileNetV2 base feature extractor was frozen (`base_model.trainable = False`).
  * Only the custom dense head (Global Average Pooling -> Dense 128 ReLU -> Dropout 0.4 -> Softmax) was trained for 5 epochs using an Adam optimizer (`lr=0.001`).
* **Phase 2 (Fine-Tuning):** 
  * The base model was unfrozen, keeping all layers **except the last 30 layers** frozen to preserve generic feature extraction while specializing the top layers for mask detection.
  * Recompiled with a micro-learning rate (`1e-5`) for delicate gradient adjustments.
  * Integrated `EarlyStopping` and `ReduceLROnPlateau` callbacks to halt training automatically at peak convergence.

---

## 📊 5. Performance & Accuracy Metrics
* **Validation Accuracy:** Achieved **100.0% validation accuracy** (`val_accuracy: 1.0000`) and near-zero validation loss (`5.3289e-04`) upon fine-tuning convergence.
* **Real-World Robustness:** Successfully eliminates false positives caused by facial shadows and unstructured lighting.

---

## 📂 6. Project Directory Structure
```text
Real_Time_FaceMask_Detection/
│
├── Face Mask Dataset/
│   ├── Train/
│   │   ├── With Mask/
│   │   └── Without Mask/
│   └── Validation/
│       ├── With Mask/
│       └── Without Mask/
│
├── mask_training_and_comparison.ipynb       # Training and evaluation notebook
├── mobilenet_finetuned_mask_model.keras     # Saved production weights artifact
├── live_inference.py                        # Real-time webcam inference script
└── README.md                                # Project documentation
