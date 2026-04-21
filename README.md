#  Gaze Estimation using Vision Transformer (ViT)

## Overview

This project predicts human gaze direction (x, y coordinates) from images using a Vision Transformer (ViT).

##  Model

* Backbone: ViT Tiny (timm)
* Transfer learning (fine-tuning last transformer block)
* Regression head for gaze prediction

##  Dataset

* GazeFollow dataset
* Contains images and gaze coordinates (normalized between 0 and 1)

##  Training

```bash
python train.py
```

##  Inference

```bash
python inference.py
```

##  Requirements

```bash
pip install -r requirements.txt
```

##  Output

* Model predicts gaze point
* Visualization using OpenCV

## 🔥 Future Work

* Add validation set
* Real-time webcam gaze tracking
* Improve accuracy with full fine-tuning
