# MLImageClassifier
Classify Types of Images using logistic regression

# Logistic Regression Digits Classifier

This project uses logistic regression to classify handwritten digits from the popular `digits` dataset in `scikit-learn`. The dataset consists of 8x8 pixel images of handwritten digits (0-9), and the goal is to classify these images using logistic regression.

## Project Overview

This project demonstrates how to:
- Load the digits dataset.
- Visualize the images.
- Train a logistic regression model to classify the digits.
- Evaluate the model's performance using accuracy and a confusion matrix.
- Visualize misclassified images.

## Dataset

The dataset is part of the `scikit-learn` library and can be loaded with `load_digits()`.

- **Features**: Each image is represented as a 64-element array (8x8 pixel values).
- **Labels**: The target labels are the digits (0-9) that correspond to the images.

```python
from sklearn.datasets import load_digits

# Load the dataset
digits = load_digits()

print("Image Data Shape", digits.data.shape)
print("Label Data Shape", digits.target.shape)
