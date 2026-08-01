# Computer Vision Portfolio & Practice Repository

A comprehensive collection of computer vision projects demonstrating practical expertise in image classification, object detection, segmentation, and generative models. Each project is self-contained with a defined scope and real-world application focus, showcasing both technical depth and versatility.

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Project Philosophy](#project-philosophy)
- [Getting Started](#getting-started)
- [How to Use](#how-to-use)

## 🎯 Overview

This repository demonstrates practical computer vision expertise through diverse, application-focused projects. Rather than following a rigid template, each project is structured based on its specific requirements, deployment target, and technical implementation.

**Repository Goals:**
- Showcase real-world problem-solving capabilities
- Practice modern CV techniques and frameworks
- Build reproducible, production-ready implementations
- Document learning and solutions comprehensively

**Each Project Provides:**
- Detailed README explaining the problem, solution, and implementation
- Reproducible environment (requirements.txt, Dockerfile, or venv setup)
- Well-organized source code with clear documentation
- Training/evaluation scripts and notebooks
- Results, metrics, and sample outputs
- Flexible architecture adaptable to different deployment scenarios

## 📁 Projects

### 1. **MNIST: Multi-Class Digit Classification** 🔤
**Status:** Active | **Difficulty:** Beginner | **Framework:** PyTorch/TensorFlow

The foundational "Hello World" of computer vision. Recognizing handwritten digits (0-9) demonstrates core ML pipeline: data loading, preprocessing, model training, and evaluation.

**Why this project:**
- Essential foundation for understanding neural networks
- Introduction to image preprocessing and normalization
- Learning evaluation metrics and validation strategies

**Key Topics:**
- Multi-class classification (10 classes: 0-9)
- Data augmentation basics
- Model architecture and training loops
- Metrics: accuracy, precision, recall, confusion matrices

**See:** `MNIST/README.md` for complete details, setup, and usage.

---

### 2. **Fashion MNIST: Fashion Item Classification** 👗
**Status:** Planned | **Difficulty:** Beginner-Intermediate

Extending digit classification to real-world objects. Classifying fashion items (t-shirt, bag, shoe, etc.) introduces working with more complex visual patterns while maintaining simplicity for learning.

**Why this project:**
- Bridge between toy datasets and real applications
- Introduction to transfer learning possibilities
- Comparison with classical ML approaches

---

### 3. **Object Detection: Real-Time Detection with YOLO** 🎯
**Status:** Planned | **Difficulty:** Intermediate | **Framework:** PyTorch + Ultralytics

Implementing state-of-the-art object detection using the latest YOLO version via Ultralytics. Detecting and localizing multiple objects in images and video streams.

**Why this project:**
- Real-world applications (surveillance, autonomous vehicles, robotics)
- Understanding bounding boxes, anchor-free detection, and NMS
- Integration with deployment frameworks

**Key Topics:**
- YOLO architecture and real-time inference
- Multi-scale feature detection
- Data augmentation for detection tasks
- Model optimization for edge deployment

---

### 4. **Semantic Segmentation: Medical Imaging** 🏥
**Status:** Planned | **Difficulty:** Intermediate-Advanced | **Framework:** PyTorch/TensorFlow

Pixel-level classification for medical image analysis. Application in analyzing medical scans (CT, MRI, X-ray) to identify and segment regions of interest.

**Why this project:**
- High-impact real-world application
- Pixel-level prediction challenges
- Working with specialized medical datasets
- Understanding encoder-decoder architectures (U-Net variants)

**Key Topics:**
- Semantic segmentation techniques
- Medical imaging preprocessing
- Class imbalance handling
- Evaluation metrics for segmentation (IoU, Dice score)

---

### 5. **Semantic Segmentation: Agricultural Applications** 🌾
**Status:** Planned | **Difficulty:** Intermediate-Advanced | **Framework:** PyTorch/TensorFlow

Precision agriculture through semantic segmentation. Analyzing satellite or drone imagery to segment crop types, detect crop health issues, and optimize resource allocation.

**Why this project:**
- Demonstrates versatility of segmentation techniques
- Remote sensing and agricultural automation
- Large-scale image processing

**Key Topics:**
- Working with multispectral imagery
- Large-scale satellite image processing
- Real-time inference for drone deployment
- Handling seasonal and environmental variations

---

### 6. **Satellite Image Segmentation: Water & Irrigation Monitoring** 💧
**Status:** Planned | **Difficulty:** Intermediate-Advanced | **Framework:** PyTorch/TensorFlow

Remote sensing and precision water management through multispectral satellite image segmentation. Analyzing Sentinel-2 and Landsat-8 imagery to segment water bodies, detect irrigation patterns, and monitor water resource utilization.

**Why this project:**
- Critical application in climate change monitoring and resource management
- Multispectral data processing and spectral index computation
- Global-scale environmental monitoring
- Practical applications in agriculture and hydrology

**Key Topics:**
- Multispectral image processing (NDWI, SWI indices)
- Working with Sentinel-2 and Landsat-8 data via Google Earth Engine
- Segmentation with deep learning (U-Net, DeepLabV3)
- Handling large-scale geospatial data
- Time-series analysis for water dynamics monitoring

**Datasets & Resources:**
- Sentinel-2 Water Segmentation dataset (GitHub public dataset)
- IRRISIGHT (irrigation-focused multimodal dataset)
- Landsat Irish Coastal Segmentation (LICS)
- RiverScope (densely labeled river segmentation)
- Global Surface Water dataset (multi-sensor Landsat-8 + Sentinel-2)

---

### 7. **CNN Explainability: Understanding Model Decisions** 🔍
**Status:** Planned | **Difficulty:** Intermediate

Making deep learning interpretable through visualization techniques. Understanding what features CNNs learn and why they make specific predictions.

**Why this project:**
- Critical for trust and deployment in sensitive applications
- Debugging model failures
- Feature visualization and attention mechanisms

**Key Topics:**
- Attention maps and gradient-based visualization
- Feature importance and activation analysis
- Class activation maps (CAM, Grad-CAM)
- Adversarial robustness analysis

---

### 8. **Style Transfer: Artistic Image Generation** 🎨
**Status:** Planned | **Difficulty:** Intermediate | **Framework:** PyTorch

Transferring artistic style from one image to another. Combining content of one image with style of another to create unique visual outputs.

**Why this project:**
- Understanding perceptual loss and content/style separation
- Creative applications of deep learning
- Working with pre-trained models and feature extraction

**Key Topics:**
- Gram matrices and style representation
- Perceptual loss functions
- Content vs. style trade-offs
- Real-time style transfer optimization

---

### 9. **Image Generation: Generative Models** 🖼️
**Status:** Planned | **Difficulty:** Advanced | **Framework:** PyTorch

Creating new images from learned distributions. Implementing generative models (VAE, GAN, Diffusion) to generate realistic images.

**Why this project:**
- Understanding generative model architectures
- Latent space exploration and manipulation
- Conditional generation for controlled outputs

**Key Topics:**
- VAE (Variational Autoencoders) architecture
- GANs (Generative Adversarial Networks)
- Diffusion models
- Training stability and convergence techniques

---

### 10. **Object Tracking: Multi-Object Tracking** 🎬
**Status:** Planned | **Difficulty:** Advanced

Tracking objects across video frames. Maintaining identity and trajectory for multiple objects in dynamic scenes.

**Why this project:**
- Video analysis and surveillance applications
- Temporal consistency and association strategies
- Real-world performance optimization

**Key Topics:**
- Detection-based tracking approaches
- Re-identification and data association
- Temporal coherence
- Video inference optimization

---

## 🏗️ Project Philosophy

### Structure Flexibility
Each project has its own structure based on specific needs:
- Some may follow a `src/` + `notebooks/` pattern
- Others might use Docker exclusively
- Some will have `config/` directories, others inline configuration
- The emphasis is on clarity and reproducibility, not rigid templates

### Technology Stack
- **Frameworks:** PyTorch, TensorFlow, JAX (project-dependent)
- **Deployment:** Local development, cloud (AWS/GCP/Azure), edge devices
- **Reproducibility:** Virtual environments, Docker, or conda specifications
- **Different Projects May Use:**
  - Different Python versions
  - Different GPU/CPU requirements
  - Different CI/CD approaches

### Integrated Techniques & Practices
Rather than standalone technique folders, concepts are integrated throughout projects:
- **Transfer Learning:** Used in projects where pre-trained models accelerate training
- **Data Augmentation:** Customized per project based on domain requirements
- **Model Optimization:** Applied during deployment phase (quantization, pruning, distillation)
- **Model Deployment:** Cloud, edge, or local based on use case
- **Experiment Tracking:** Most projects integrate observability tools for tracking experiments, models, and parameters

### Experiment Tracking & Observability

Most projects include experiment tracking to monitor training progress, compare models, and maintain reproducibility:

**MLflow**
- Lightweight, open-source experiment tracking
- Logs parameters, metrics, and artifacts
- Model registry for version control
- Easy integration with existing workflows
- Suitable for local and self-hosted setups

**Weights & Biases (W&B)**
- Cloud-based experiment tracking platform
- Real-time visualization and collaboration
- Hyperparameter sweeps and optimization
- Model versioning and artifact management
- Rich dashboards and reporting capabilities

**What Gets Tracked:**
- Training/validation metrics (loss, accuracy, etc.)
- Hyperparameters and model configuration
- Model artifacts and checkpoints
- Dataset information and preprocessing details
- Training duration and hardware usage
- Visualizations and sample predictions

Each project's README specifies which observability tool is used and how to access/configure it.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- pip or conda

### General Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cv_portfolio
```

2. **Navigate to a project:**
```bash
cd MNIST
```

3. **Follow project-specific setup** (consult `README.md` in each project folder for exact instructions, as they vary):
```bash
# Project-specific setup examples:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Or with Docker:
docker build -t project-name .
docker run -it project-name
```

## 💻 How to Use

### Exploring Projects
1. Start with **MNIST** - simplest foundation
2. Read each project's README for context: Why? What? How?
3. Review source code structure and comments
4. Check notebooks for step-by-step walkthroughs
5. Experiment with hyperparameters and data

### Running Projects
Each project's README contains specific instructions for:
- Data acquisition/preparation
- Environment setup
- Training procedures
- Evaluation and testing
- Generating predictions

### Adapting Projects
- Understand the core problem and solution approach
- Modify for your own datasets or use cases
- Integrate techniques into your own projects
- Reference implementations as examples

## 📚 Learning Path

**Suggested progression:**

1. **Foundation** → MNIST (core ML pipeline)
2. **Classification** → Fashion MNIST (more complex patterns)
3. **Detection** → Object Detection (localization, multiple objects)
4. **Segmentation** → Medical or Agro (pixel-level prediction)
5. **Understanding** → CNN Explainability (model interpretation)
6. **Advanced** → Style Transfer, Generation, Tracking

## 🔮 Planned Projects Checklist

- [ ] Fashion MNIST - Multi-class classification
- [ ] Object Detection (YOLO) - Real-time detection
- [ ] Semantic Segmentation (Medical) - Healthcare applications
- [ ] Semantic Segmentation (Agricultural) - Remote sensing
- [ ] Satellite Image Segmentation (Water/Irrigation) - Geospatial analysis
- [ ] CNN Explainability - Model interpretation
- [ ] Style Transfer - Artistic generation
- [ ] Image Generation - Generative models
- [ ] Object Tracking - Video analysis

## 📝 Project Documentation

Each project includes:
- **README.md** - Problem statement, solution approach, setup instructions
- **Source code** - Well-commented, modular implementation
- **Notebooks** - Interactive exploration and results analysis
- **Configuration** - Hyperparameters and settings (format varies per project)
- **Results** - Metrics, visualizations, and sample outputs

## 🎓 Key Topics Across Projects

- Image classification and multi-class problems
- Object detection and localization
- Semantic and instance segmentation
- Generative models and style transfer
- Model interpretability and explainability
- Real-time inference optimization
- Transfer learning and fine-tuning
- Data augmentation strategies
- Loss functions and metrics
- Experiment tracking and observability (MLflow, Weights & Biases)
- Deployment and productionization

## 🛠️ Notes

- **Hardware:** Projects document CPU/GPU requirements and training times
- **Reproducibility:** Random seeds and configuration files ensure consistent results
- **Versions:** Each project specifies framework versions in requirements
- **Scalability:** Code examples demonstrate both research and production patterns

---

**Start with MNIST, explore at your own pace, and learn by doing!** 🚀

For detailed information about any specific project, see its individual README.md file.
