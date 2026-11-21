# Sign Language Recognition (ASL alphabet)

Classifies 24 static American Sign Language letters from 28×28 grayscale hand images, comparing an SVM, a tuned MLP and a CNN. Every run is tracked in Weights & Biases.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paldenzimbatamang/asl-sign-language-recognition/blob/main/Final_tensorflow_wandb.ipynb)

## Results (held-out test set, 3,586 images)

| Model | Test accuracy | Macro F1 |
| --- | --- | --- |
| CNN (3 conv layers, 64 filters, dropout 0.2) | 90.41% | 0.8955 |
| SVM (RBF, C=10), week 1 baseline | 83.55% | 0.8188 |
| MLP (best of 9 settings: 1024-512-256, ReLU) | 82.99% | 0.8135 |

## What I did

- Built the SVM baseline on my own in week 1, giving the team a benchmark for later models
- Designed the CNN and deployed it as a Flask web app that classifies uploaded hand-sign photos
- Ran 9 MLP experiments (depth, width, activation, learning rate) with early stopping, logging every run to W&B
- Found a data leakage problem: the training file contains augmented near-copies of each photo, so a random validation split scored 100% while real test accuracy was 79%. I moved validation to half of the official test file and kept the other half for a single final evaluation

This was a team of 3. A teammate built the Random Forest model.

## Data

[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) on Kaggle: 27,455 training images and 7,172 test images. J and Z are excluded because they need motion.

## Run it

Click the Colab badge above, or locally:

```
pip install kagglehub tensorflow scikit-learn matplotlib pandas wandb
jupyter notebook Final_tensorflow_wandb.ipynb
```

The notebook downloads the dataset with kagglehub and asks for a W&B API key (choose "Don't visualize" to skip logging).

## Tech

Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, matplotlib, Weights & Biases
