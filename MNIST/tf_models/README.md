# TensorFlow Implementations

This directory contains TensorFlow-based implementations of MNIST digit classification models.

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
python fcnn/train.py

# CNN Training
python cnn/train.py
```

## Key Features

- **Keras Sequential API**: Clean and readable model definitions
- **Dataset Pipeline**: Efficient tf.data.Dataset for data loading
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Metrics Tracking**: Comprehensive evaluation metrics
- **Reproducibility**: Fixed seeds for deterministic results

## Documentation

See individual README files in each model directory for architecture details and training hyperparameters.
