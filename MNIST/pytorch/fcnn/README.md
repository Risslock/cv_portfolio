# PyTorch Fully Connected Neural Network

A production-grade implementation of a fully connected neural network for MNIST digit classification using PyTorch.

## Architecture

**Model Structure**:
- Input Layer: 784 neurons (28×28 flattened images)
- Hidden Layer 1: 128 neurons + ReLU activation + Dropout(0.2)
- Hidden Layer 2: 64 neurons + ReLU activation + Dropout(0.2)
- Output Layer: 10 neurons + LogSoftmax activation

**Loss Function**: Cross-Entropy Loss (NLLLoss)

**Optimizer**: Adam (learning rate: 0.001)

## Key Components

### `model.py`
Defines the `FCNNModel` class extending `torch.nn.Module`:
```python
class FCNNModel(nn.Module):
    """Fully connected neural network for MNIST classification."""
```

### `train.py`
Complete training pipeline:
- Data loading with PyTorch DataLoaders
- Model instantiation and training loop
- Validation during training
- Model evaluation and metrics computation
- Checkpoint saving to `../results/pytorch_fcnn/`

## Training

```bash
python -m pytorch.fcnn.train
```

**Configuration** (modify in `train.py`):
- Epochs: 20 (default)
- Batch Size: 32
- Learning Rate: 0.001
- Device: CPU/CUDA (automatic detection)

## Expected Results

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~97-98%
- **Training Time**: ~10-15 seconds on modern CPU
- **GPU Acceleration**: ~5-8 seconds on modern GPU

## Model Output

Training saves:
- `model_best.pth` - Best model state_dict
- `model_final.pth` - Final model state_dict
- `metrics.json` - Training/validation metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Loss and accuracy plots

## Usage

```python
import torch
from pytorch.fcnn.model import FCNNModel
from torchvision import datasets, transforms

# Load and preprocess data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, 
                               download=True, transform=transform)

# Create model
model = FCNNModel(input_size=784, hidden_sizes=[128, 64], num_classes=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

## Jupyter Notebook

See `../notebooks/pytorch_fcnn_walkthrough.ipynb` for an interactive walkthrough of:
1. Dataset downloading and preprocessing
2. DataLoader creation
3. Model building with nn.Module
4. Training loop implementation
5. Evaluation and metrics
6. Visualization

## PyTorch-Specific Features

- **Computational Graphs**: Dynamic computation graph for debugging
- **GPU Support**: Seamless CUDA acceleration
- **State Dictionary**: Easy model serialization with state_dict
- **autograd**: Automatic differentiation
- **TorchVision**: Pre-built datasets and transforms

## Comparison with TensorFlow

| Aspect | PyTorch | TensorFlow |
|--------|---------|-----------|
| API Style | Pythonic, imperative | Declarative (Keras) |
| Debugging | Easier (dynamic graphs) | Harder (static graphs) |
| Performance | Comparable | Comparable |
| Community | Research-focused | Production-focused |
| GPU Support | Native CUDA | TensorRT optimization |

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- PyTorch nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- TorchVision: https://pytorch.org/vision/stable/index.html
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
