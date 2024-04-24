# README: Deep Neural Network with TensorFlow

## Introduction

This project demonstrates the development and training of a deep neural network using TensorFlow to classify handwritten digits from the MNIST dataset. The MNIST dataset comprises 28x28 pixel grayscale images of handwritten digits (0-9), making it one of the most used datasets for benchmarking image processing systems.

## Project Setup

### Dependencies

- TensorFlow 2.12.0
- TensorFlow Datasets 4.9.2
- TensorFlow Estimator 2.12.0
- Other relevant TensorFlow libraries as listed in the project requirements.

To install a specific version of TensorFlow, use:
```bash
pip install tensorflow==2.0.0-beta0
```

### Data

We use the MNIST dataset which is directly loaded via TensorFlow's dataset library. The images are 28x28 pixels and the dataset is split into training and test sets.

## Model Architecture

This project implements a multi-layer perceptron with the following architecture:

- **Input Layer**: 784 neurons (flattened from 28x28)
- **First Hidden Layer**: 128 neurons, ReLU activation, Batch Normalization
- **Second Hidden Layer**: 64 neurons, ReLU activation, Batch Normalization
- **Third Hidden Layer**: 64 neurons, ReLU activation, Batch Normalization, Dropout (0.2)
- **Output Layer**: 10 neurons (corresponding to the 10 digits), Softmax activation

## Preprocessing

Input images are reshaped from 28x28 matrices to 784-element vectors and normalized to have values between 0 and 1.

## Training

The model is compiled with:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: NADAM
- **Metrics**: Accuracy

It is trained over 10 epochs with a batch size of 128. Validation is performed using the test set.

## Performance

After training, the model achieves approximately 97.24% accuracy on the test set. Model performance metrics are detailed at the end of each training epoch.

## Usage

To predict new data points, reshape the input data to conform to the model's input structure (1x784) and pass it to the model's `predict` method.


