# PyTorch Convolutional Neural Network

A production-grade implementation of a convolutional neural network for MNIST digit classification using PyTorch.

## Architecture

**Model Structure**:
- Conv Layer 1: 32 filters, 3×3 kernel + ReLU + MaxPool(2×2)
- Conv Layer 2: 64 filters, 3×3 kernel + ReLU + MaxPool(2×2)
- Flatten
- Dense Layer: 128 neurons + ReLU + Dropout(0.5)
- Output Layer: 10 neurons + LogSoftmax activation

**Loss Function**: Cross-Entropy Loss (NLLLoss)

**Optimizer**: Adam (learning rate: 0.001)

## Key Components

### `model.py`
Defines the `CNNModel` class extending `torch.nn.Module`:
```python
class CNNModel(nn.Module):
    """Convolutional neural network for MNIST classification."""
```

### `train.py`
Complete training pipeline:
- Data loading preserving spatial structure (28×28×1)
- Model instantiation and training loop
- Validation during training
- Model evaluation with confusion matrix
- Checkpoint saving and result visualization

## Training

```bash
python -m pytorch.cnn.train
```

**Configuration** (modify in `train.py`):
- Epochs: 15 (default)
- Batch Size: 32
- Learning Rate: 0.001
- Device: CPU/CUDA (automatic detection)

## Expected Results

- **Training Accuracy**: ~99.5%
- **Test Accuracy**: ~99-99.5%
- **Training Time**: ~20-30 seconds on modern CPU
- **GPU Acceleration**: ~10-15 seconds on modern GPU

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
from pytorch.cnn.model import CNNModel
from torchvision import datasets, transforms

# Load and preprocess data (keep spatial structure)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)

# Create model
model = CNNModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # Input shape: (batch_size, 1, 28, 28)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

## Jupyter Notebook

See `../notebooks/pytorch_cnn_walkthrough.ipynb` for an interactive walkthrough of:
1. Dataset preparation with spatial preservation
2. Convolutional layer design
3. Training loop with progress tracking
4. Filter visualization
5. Performance analysis and comparison

## Key Differences from FCNN

- **Spatial Information**: Preserves 2D structure (1×28×28)
- **Feature Learning**: Automatic feature extraction via convolutions
- **Parameter Efficiency**: ~500K parameters vs ~1M for FCNN
- **Performance**: ~2% accuracy improvement
- **Interpretability**: Visualizable convolutional filters

## PyTorch-Specific Patterns

### Model Definition Pattern
```python
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # ... more layers
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # ... forward pass
        return x
```

### Training Loop Pattern
```python
model.train()
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

## Visualization & Analysis

The training script includes:
- Real-time loss/accuracy tracking
- Confusion matrix generation
- Training curves plotting
- Per-class performance metrics

## References

- PyTorch Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- CNN Basics: http://cs231n.github.io/convolutional-networks/
- PyTorch Examples: https://github.com/pytorch/examples
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
