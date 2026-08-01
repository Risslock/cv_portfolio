# MNIST Digit Classification: ML & Software Engineering Showcase

A comprehensive machine learning project demonstrating expertise in neural network architectures, multiple deep learning frameworks, and professional software development practices.

## Project Overview

This project implements digit classification using the MNIST dataset (28×28 grayscale images, 10 classes) with multiple framework and architecture combinations:

- **TensorFlow**: Fully Connected Neural Networks (FCNN) & Convolutional Neural Networks (CNN)
- **PyTorch**: Fully Connected Neural Networks (FCNN) & Convolutional Neural Networks (CNN)

Each implementation showcases production-grade code organization, comprehensive documentation, and systematic evaluation.

## Quick Start

### Prerequisites

- Python 3.10+
- UV (Python package manager)

### Installation

1. **Clone and navigate to the project**:
   ```bash
   cd MNIST
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv sync
   ```

3. **Verify setup**:
   ```bash
   python -c "import tensorflow; import torch; print('Setup successful!')"
   ```

### Running Models

Each framework/model combination has a training script in its subdirectory:

```bash
# TensorFlow FCNN
python tensorflow/fcnn/train.py

# TensorFlow CNN
python tensorflow/cnn/train.py

# PyTorch FCNN
python pytorch/fcnn/train.py

# PyTorch CNN
python pytorch/cnn/train.py
```

### Exploring Notebooks

Jupyter notebooks provide step-by-step walkthroughs:

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `tensorflow_fcnn_walkthrough.ipynb` - Data loading, preprocessing, model building, training
- `tensorflow_cnn_walkthrough.ipynb` - CNN architecture and training pipeline
- `pytorch_fcnn_walkthrough.ipynb` - PyTorch equivalent of FCNN
- `pytorch_cnn_walkthrough.ipynb` - PyTorch CNN implementation

## Project Structure

```
MNIST/
├── tensorflow/              # TensorFlow implementations
│   ├── fcnn/               # Fully Connected Neural Network
│   │   ├── __init__.py
│   │   ├── model.py        # Model class definition
│   │   ├── train.py        # Training pipeline
│   │   └── README.md       # Framework-specific documentation
│   └── cnn/                # Convolutional Neural Network
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── README.md
│
├── pytorch/                 # PyTorch implementations
│   ├── fcnn/               # Fully Connected Neural Network
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── README.md
│   └── cnn/                # Convolutional Neural Network
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── README.md
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── tensorflow_fcnn_walkthrough.ipynb
│   ├── tensorflow_cnn_walkthrough.ipynb
│   ├── pytorch_fcnn_walkthrough.ipynb
│   └── pytorch_cnn_walkthrough.ipynb
│
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── metrics.py          # Evaluation metrics and visualization
│   └── visualization.py    # Plot utilities
│
├── data/                   # MNIST dataset (auto-downloaded)
│
├── results/                # Training outputs and models
│   ├── tensorflow_fcnn/    # Saved models, metrics, logs
│   ├── tensorflow_cnn/
│   ├── pytorch_fcnn/
│   └── pytorch_cnn/
│
├── pyproject.toml          # Project dependencies (UV)
└── README.md               # This file
```

## Key Features

### 1. Object-Oriented Design
All code follows OOP principles:
- Model classes with clear interfaces
- Reusable data loader utilities
- Modular evaluation pipelines
- No procedural scripts

### 2. Professional Code Quality
- **PEP 8 Compliance**: All code follows Python style guidelines
- **Linting**: Passes flake8 and pylint checks
- **Type Hints**: Annotated for improved clarity
- **Docstrings**: Complete NumPy-style docstrings for all public functions

### 3. Comprehensive Documentation
- Detailed docstrings for all classes and functions
- Usage examples in code comments
- Jupyter notebooks demonstrating full pipelines
- Framework-specific READMEs in each subdirectory

### 4. Reproducible Workflows
- Jupyter notebooks show step-by-step processes
- Seed management for reproducibility
- Clear data preprocessing pipeline
- Documented hyperparameters

## Code Standards

### Folder Organization
Each framework/model combination follows this pattern:
- `model.py` - Model class definition(s)
- `train.py` - Training script with experiment tracking
- `__init__.py` - Package initialization
- `README.md` - Implementation-specific notes

### Naming Conventions
- Classes: `PascalCase` (e.g., `FCNNModel`, `CNNModel`)
- Functions: `snake_case` (e.g., `load_data`, `evaluate_model`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `BATCH_SIZE`, `EPOCHS`)

### Docstring Format
All public functions and classes include complete docstrings:

```python
def train_model(model, train_loader, epochs=10, lr=0.001):
    """
    Train a neural network model on the MNIST training dataset.
    
    Args:
        model: Neural network model instance
        train_loader: PyTorch DataLoader for training data
        epochs (int): Number of training epochs (default: 10)
        lr (float): Learning rate (default: 0.001)
    
    Returns:
        dict: Training history with loss values per epoch
    
    Raises:
        ValueError: If epochs < 1 or lr <= 0
    
    Example:
        history = train_model(model, train_loader, epochs=20, lr=0.0001)
    """
```

## Model Architectures

### Fully Connected Neural Network (FCNN)
- Input layer: 784 neurons (28×28 flattened images)
- Hidden layers: 128 → 64 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation
- **Purpose**: Baseline architecture demonstrating fundamental concepts

### Convolutional Neural Network (CNN)
- Conv1: 32 filters, 3×3 kernel + ReLU + MaxPool
- Conv2: 64 filters, 3×3 kernel + ReLU + MaxPool
- Dense: 128 neurons + ReLU + Dropout(0.5)
- Output: 10 neurons with softmax activation
- **Purpose**: Demonstrates spatial feature learning and modern best practices

## Dependencies

Managed via UV in `pyproject.toml`:

**Core Libraries**:
- numpy - Numerical computations
- tensorflow - TensorFlow framework
- torch - PyTorch framework
- torchvision - PyTorch computer vision utilities
- scikit-learn - Metrics and utilities

**Development**:
- jupyter - Notebook environment
- matplotlib - Visualization
- seaborn - Enhanced plots
- flake8 - Code linting
- pylint - Advanced linting
- black - Code formatting

## Usage Examples

### Using the Model Classes Directly

```python
from pytorch.fcnn.model import FCNNModel
from utils.data_loader import load_mnist_data

# Load data
train_loader, test_loader = load_mnist_data(batch_size=32)

# Create model
model = FCNNModel(input_size=784, hidden_sizes=[128, 64], num_classes=10)

# Train (see train.py for full example)
model.train()
```

### Evaluating Models

```python
from utils.metrics import evaluate_model, plot_confusion_matrix

metrics = evaluate_model(model, test_loader)
print(f"Accuracy: {metrics['accuracy']:.4f}")
plot_confusion_matrix(metrics['confusion_matrix'], save_path='results/confusion.png')
```

## Training & Evaluation

Each implementation includes:

1. **Data Loading**: MNIST auto-download and preprocessing
2. **Training Loop**: Epoch-based training with validation
3. **Metrics Tracking**: Accuracy, loss, confusion matrix
4. **Model Checkpointing**: Save best model by validation accuracy
5. **Visualization**: Training curves and confusion matrices

Expected performance:
- **FCNN**: ~97% accuracy on MNIST test set
- **CNN**: ~99% accuracy on MNIST test set

## Project Goals

This project demonstrates:

✅ **Machine Learning Expertise**:
- Deep understanding of neural network architectures
- Practical experience with two major frameworks
- Ability to implement and train models from scratch

✅ **Software Engineering Excellence**:
- Production-grade code organization
- Professional documentation standards
- Reproducible workflows and experiments
- Clear API design and reusable components

✅ **Communication Skills**:
- Well-structured codebase self-documents its purpose
- Comprehensive README and inline documentation
- Jupyter notebooks explain methodology clearly
- Clear folder hierarchy aids understanding

## Getting Help

Each subdirectory has a `README.md` with framework-specific details.

For general questions, review:
1. The relevant notebook for that framework/model
2. Docstrings in the model and utility files
3. Framework-specific README in the subdirectory

## Contributing

This project follows the MNIST Constitution (`.specify/memory/constitution.md`). All code must:
- Follow OOP principles with clear class hierarchies
- Maintain easy-to-understand folder structure
- Include complete docstrings and type hints
- Pass PEP 8 style checks (flake8, pylint)
- Include Jupyter notebooks for complex implementations

## License

Personal portfolio project - Educational Use

---

**Created**: 2026-07-31  
**Status**: Active Development  
**Python Version**: 3.10+
