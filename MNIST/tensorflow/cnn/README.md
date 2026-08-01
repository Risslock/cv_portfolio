# TensorFlow Convolutional Neural Network

A production-grade implementation of a convolutional neural network for MNIST digit classification using TensorFlow/Keras.

## Architecture

**Model Structure**:
- Conv Layer 1: 32 filters, 3×3 kernel + ReLU + MaxPool(2×2)
- Conv Layer 2: 64 filters, 3×3 kernel + ReLU + MaxPool(2×2)
- Flatten
- Dense Layer: 128 neurons + ReLU + Dropout(0.5)
- Output Layer: 10 neurons + Softmax activation

**Regularization**:
- Dropout (0.5) before output
- Early stopping on validation loss
- L2 regularization on dense layers

## Key Components

### `model.py`
Defines the `CNNModel` class:
```python
class CNNModel(tf.keras.Model):
    """Convolutional neural network for MNIST classification."""
```

### `train.py`
Complete training pipeline:
- Data loading and preprocessing (keeps spatial structure)
- Model instantiation and compilation
- Training loop with callbacks
- Model evaluation with confusion matrix
- Result visualization and saving

## Training

```bash
python train.py
```

**Configuration** (modify in `train.py`):
- Epochs: 15 (default)
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: Adam

## Expected Results

- **Training Accuracy**: ~99.5%
- **Test Accuracy**: ~99-99.5%
- **Training Time**: ~20-30 seconds on modern CPU

## Model Output

Training saves:
- `model.h5` - Trained model weights
- `metrics.json` - Training/validation metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Loss and accuracy plots

## Usage

```python
from tensorflow.cnn.model import CNNModel
from tensorflow.keras.datasets import mnist

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Create model
model = CNNModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.1)

# Evaluate
model.evaluate(x_test, y_test)
```

## Jupyter Notebook

See `../notebooks/tensorflow_cnn_walkthrough.ipynb` for an interactive walkthrough of:
1. Data loading with spatial structure preservation
2. Convolutional architecture design
3. Training with visualization
4. Feature map visualization
5. Performance analysis

## Key Differences from FCNN

- **Spatial Information**: Preserves 2D structure of images
- **Feature Learning**: Automatic feature extraction via convolutions
- **Efficiency**: Fewer parameters than equivalent FCNN
- **Performance**: Higher accuracy (~2% improvement)
- **Interpretability**: Can visualize learned filters

## References

- TensorFlow Convolution: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
- CNN Fundamentals: http://cs231n.github.io/convolutional-networks/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
