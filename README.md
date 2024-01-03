
# ElasticTorch README

This package is designed to facilitate quick benchmarks comparing CPU and GPU performance for ElasticNet models using PyTorch. The decision to use PyTorch is rooted in its flexibility to move operations between CPU and GPU seamlessly, and its capabilities that extend beyond a mere deep learning package.

## Overview of ElasticTorch

ElasticTorch is a Python package that utilizes the power of PyTorch for numerical computations and machine learning. It focuses on providing tools for ElasticNet regularization, a popular method in regression models that combines L1 and L2 penalties. This approach is particularly beneficial for feature selection and preventing overfitting.

## Motivation

The primary purpose of developing ElasticTorch is to support benchmarks between CPU and GPU performances specifically for ElasticNet models. Understanding how your hardware can influence training times and model performance is vital, especially for complex models and large datasets.

## Getting Started

Ensure you have PyTorch installed on your system, along with appropriate CUDA versions if using GPU. If not, please visit the [PyTorch official website](https://pytorch.org/get-started/locally/) and follow the installation instructions.

### ElasticNet Model

The `ElasticNet` class is a PyTorch module designed for regression with ElasticNet regularization. Below is a quick guide on its instantiation and usage:

```python
import torch
from elastictorch import ElasticNet

# Number of features in your dataset
n_features = 10

# Initialize the ElasticNet model
model = ElasticNet(n_features=n_features)

# Sample data
X = torch.randn((100, n_features))
y = torch.randn(100, 1)

# Forward pass
predictions = model(X)

# Compute the loss
loss = model.loss(predictions, y)
print(f"Loss: {loss.item()}")
```

### Training with PyTorch Optimizer Trainer

The `PyTorchOptimizerTrainer` handles the training loop. It's designed to simplify the training process:

```python
from elastictorch import ElasticNet, PyTorchOptimizerTrainer
import torch.optim as optim

# Assuming model initialization
optimizer = optim.Adam(model.parameters())

# Initialize the trainer
trainer = PyTorchOptimizerTrainer(model, optimizer)

# Train the model for a specified number of epochs
trainer.train(X_train, y_train, epochs=100)
```

### Running Benchmarks

The `make_test.py` script is designed to facilitate benchmarking. It allows you to specify the device, number of samples, features, and other parameters:

```bash
python make_test.py --device cuda --n_samples 10000 --n_features 20
```

This command runs the model on the GPU with 10,000 samples and 20 features each.

Set `device` to 'cpu' to run on CPU, and to 'cuda' to run on GPU. See the `benchmark.ipynb` notebook for an example comparison of CPU and GPU performance. Note, the advantage of GPU scales with data size.
