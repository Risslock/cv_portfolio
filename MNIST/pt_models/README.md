# PyTorch Implementations

This directory contains PyTorch-based implementations of MNIST digit classification models.

## Structure

- `fcnn/` - Fully Connected Neural Network
- `cnn/` - Convolutional Neural Network

## Getting Started

Each subdirectory (fcnn, cnn) contains:
- `model.py` - Model class definition
- `train.py` - Training script with experiment tracking
- `README.md` - Implementation-specific documentation

## Running Models

```bash
# FCNN Training
python -m pytorch.fcnn.train

# CNN Training
python -m pytorch.cnn.train
```

## Key Features

- **nn.Module Architecture**: PyTorch's standard neural network interface
- **DataLoader Pipeline**: PyTorch DataLoaders for efficient batching
- **Training Loops**: Explicit epoch-based training for clarity
- **Model Persistence**: Save/load functionality with state_dict
- **GPU Support**: CUDA acceleration when available
- **Metrics Tracking**: Comprehensive evaluation and visualization

## Documentation

See individual README files in each model directory for architecture details and training hyperparameters.
