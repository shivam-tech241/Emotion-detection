# Emotion Detection

Detects human emotions from facial expressions using deep learning.
Compares a custom CNN against pretrained models (VGG16, ResNet50V2).

## Models Compared
| Model      | Accuracy |
|------------|----------|
| Custom CNN | 64.14%   |
| VGG16      | 37.30%   |
| ResNet50V2 | 46.28%   |

**Best Model: Custom CNN (64.14%)**

## Key Finding
Custom CNN outperformed pretrained models (VGG16, ResNet50V2) because
FER2013 uses small 48×48 grayscale images, while pretrained models were
trained on large RGB ImageNet images.

## Model Architecture
- **CNN:** 10-layer custom architecture with BatchNorm + Dropout
- **VGG16:** Transfer learning with frozen ImageNet weights
- **ResNet50V2:** Partial fine-tuning (last 20 layers unfrozen)

## Highlights
- Data augmentation (rotation, zoom, flip, brightness)
- EarlyStopping + ModelCheckpoint callbacks
- 6 emotion classes: angry, fear, happy, neutral, sad, surprise

## Tech Stack
Python, TensorFlow, Keras, NumPy, Matplotlib

## Dataset
FER2013 — Facial Expression Recognition
[Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

## How to Run
1. Upload dataset to Google Drive
2. Open notebook in Google Colab
3. Mount Drive and run all cells
