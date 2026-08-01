# MNIST CNN Learning Guides

This directory contains Jupyter notebooks that serve as comprehensive learning guides for implementing Convolutional Neural Networks (CNNs) on the MNIST digit classification task using both TensorFlow and PyTorch.

## Notebooks

### 1. **tensorflow_cnn_guide.ipynb**
Complete walkthrough of CNN implementation using TensorFlow/Keras.

**Contents:**
- Data loading and visualization
- Data preprocessing (normalization, reshaping)
- CNN model definition with architecture explanation
- Training setup with MLflow experiment tracking
- Training loop with validation monitoring
- Evaluation and metrics computation
- Model persistence and artifact logging

**Learning Goals:**
- Understand TensorFlow Sequential API
- Learn how convolution and pooling extract spatial features
- Implement training callbacks for metric logging
- Integrate MLflow for reproducible experiments

**Execution Time:** <2 minutes on CPU

### 2. **pytorch_cnn_guide.ipynb**
Parallel implementation using PyTorch with emphasis on framework differences.

**Contents:**
- MNIST data loading with PyTorch DataLoaders
- NCHW tensor format (channels-first) explanation
- CNN model defined as torch.nn.Module subclass
- Custom training loop with optimizer and backward pass
- Evaluation with context managers (`torch.no_grad()`)
- Model serialization and MLflow integration

**Learning Goals:**
- Understand PyTorch's dynamic computation graph
- Learn differences between TensorFlow and PyTorch APIs
- Implement custom training loops for fine-grained control
- Compare framework performance via MLflow

**Execution Time:** <2 minutes on CPU

## Running Notebooks

### Prerequisites
Ensure dependencies are installed:
```bash
uv pip install tensorflow pytorch jupyter matplotlib scikit-learn mlflow
```

### Launch Jupyter
```bash
cd notebooks/
uv run jupyter notebook
```

Then open either notebook in the Jupyter interface and execute cells sequentially.

### Quick Start (TensorFlow)
```bash
# In notebook or Python terminal
%run tensorflow_cnn_guide.ipynb
```

### Quick Start (PyTorch)
```bash
# In notebook or Python terminal
%run pytorch_cnn_guide.ipynb
```

## Framework Comparison

Both notebooks implement the same CNN architecture for direct comparison:

| Aspect | TensorFlow | PyTorch |
|--------|-----------|---------|
| **API Style** | High-level (Sequential) | Low-level (Module) |
| **Training** | model.fit() | Custom loop |
| **Data Format** | NHWC (channels-last) | NCHW (channels-first) |
| **Output** | Softmax probabilities | Logits (raw scores) |
| **Typical Use** | Production pipelines | Research & experimentation |

## Viewing Results in MLflow

After running a notebook, view experiments and metrics:

```bash
mlflow ui --port 5000
```

Then navigate to:
- http://localhost:5000/
- Select experiment: `cnn_mnist_tensorflow` or `cnn_mnist_pytorch`
- Compare metrics, parameters, and artifacts

## Metrics Tracked

Both notebooks log:

**Parameters (11 total):**
- learning_rate, batch_size, num_epochs
- optimizer, loss_function
- num_conv_blocks, conv_filters_initial, dense_units, dropout_rate
- random_seed (always 42), framework

**Per-Epoch Metrics:**
- train_loss, train_accuracy, val_loss, val_accuracy

**Final Test Metrics:**
- test_loss, test_accuracy, test_precision, test_recall, test_f1
- training_time_seconds, inference_time_ms_per_batch

## Architecture Details

Both implementations use:
- **2 Convolutional Blocks**: Extract spatial features at different scales
  - Block 1: 32 filters (28×28 → 14×14 after pooling)
  - Block 2: 64 filters (14×14 → 7×7 after pooling)
- **Dense Layers**: Classification
  - Hidden layer: 128 units with ReLU and 50% dropout
  - Output layer: 10 units (one per digit class)

## Expected Performance

- **Test Accuracy:** >98% (validates CNN effectiveness)
- **Training Time:** <2 minutes per notebook on CPU
- **Inference Time:** <50ms per batch on CPU

## Next Steps

After understanding these notebooks:
1. Review `tensorflow/cnn/train.py` and `pytorch/cnn/train.py` for production training scripts
2. Check `tensorflow/cnn/models.py` and `pytorch/cnn/models.py` for reusable model classes
3. Compare MLflow experiments to understand framework trade-offs

## References

- [TensorFlow Keras Documentation](https://www.tensorflow.org/api/keras)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [CNN Architecture Basics](https://cs231n.github.io/convolutional-networks/)

---

**Created:** 2026-08-01  
**Last Updated:** 2026-08-01  
**Constitutional Principle:** IV (Jupyter Notebooks for Learning)
