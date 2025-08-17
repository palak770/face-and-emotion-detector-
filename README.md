# Face and Emotion Detector

This project detects human emotions from facial images using **VGG16-based CNN**.

## Features
- Uses pre-trained VGG16 for feature extraction.
- Fine-tunes on your custom dataset.
- Supports multiple emotion classes.
- Training and testing scripts included.

## Folder Structure



Face-Emotion-Detector/
├─ .env/    # Virtual environment (ignored)

├─ train/   # Training dataset (ignored)

├─ test/    # Testing dataset (ignored)

├─ emotion_detector.py   # Main detection code

├─ train_model.py    # Training script

├─ testing.py   # Testing script

├─ my_custom_emotion_model.h5 # Saved model (ignored)

├─ README.md

└─ .gitignore




## Requirements
```bash
pip install tensorflow keras numpy opencv-python


