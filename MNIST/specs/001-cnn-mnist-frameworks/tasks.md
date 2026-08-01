# Tasks: CNN MNIST Frameworks

**Input**: Design documents from `/specs/001-cnn-mnist-frameworks/`

**Prerequisites**: plan.md (required), spec.md (user stories), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Unit and integration tests included (per specification requirement FR-012, FR-013)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. All user stories are P1/P2 priority; recommended MVP includes US1 + US2.

## Format: `- [ ] [ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Project root**: Repository root where pyproject.toml exists
- **Framework dirs**: `tensorflow/cnn/`, `pytorch/cnn/`
- **Notebooks**: `notebooks/`
- **Tests**: `tests/`
- **Utilities**: `utils/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure per constitution principles

- [ ] T001 Create directory structure: tensorflow/cnn/, pytorch/cnn/, notebooks/, tests/, utils/
- [ ] T002 Update pyproject.toml with TensorFlow, PyTorch, MLflow, Jupyter, pytest dependencies
- [ ] T003 [P] Create tensorflow/cnn/__init__.py and pytorch/cnn/__init__.py files
- [ ] T004 [P] Create .gitignore entries for mlruns/, *.pyc, __pycache__, .ipynb_checkpoints

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure required by ALL user stories before any CNN implementation

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Create utils/mlflow_config.py with MLflow experiment setup function (uses seed=42, consistent naming)
- [ ] T006 Create utils/metrics.py with accuracy, precision, recall, F1 computation utilities
- [ ] T007 [P] Create tensorflow/cnn/data.py: MNIST data loading, preprocessing, train/val/test split
- [ ] T008 [P] Create pytorch/cnn/data.py: MNIST data loading, preprocessing, DataLoader creation
- [ ] T009 Create utils/__init__.py with utility exports
- [ ] T010 [P] Create tests/__init__.py and tests/conftest.py with pytest fixtures (seed=42, temp directories)
- [ ] T011 Validate data loading: unit test in tests/test_data_loading.py (both frameworks load identical MNIST splits)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Framework Comparison via MLflow (Priority: P1) 🎯 MVP

**Goal**: Implement CNN models for TensorFlow and PyTorch with MLflow tracking enabled, enabling framework comparison through consistent parameter/metric logging.

**Independent Test**: Both models train with MLflow experiment tracking; metrics visible in MLflow UI; parameter naming identical across frameworks; test accuracy ≥98% for both.

### Implementation for User Story 1

- [ ] T012 [P] [US1] Create tensorflow/cnn/models.py: MNISTCNNModel class (2 conv blocks, seed=42 support)
  - Inherits keras.Sequential
  - Constructor: input_shape=(28,28,1), num_classes=10, conv_filters_initial=32, dense_units=128, dropout_rate=0.5, seed=42
  - Includes complete docstring per Principle V
  
- [ ] T013 [P] [US1] Create pytorch/cnn/models.py: MNISTCNNModel class (2 conv blocks, seed=42 support)
  - Inherits torch.nn.Module
  - Constructor: input_channels=1, num_classes=10, conv_filters_initial=32, dense_units=128, dropout_rate=0.5, seed=42
  - Includes complete docstring per Principle V
  - Verify architectural equivalence to TensorFlow model (contracts/model-interface.md)

- [ ] T014 [P] [US1] Create tensorflow/cnn/train.py: Training script with MLflow logging
  - Log parameters: learning_rate=0.001, batch_size=32, num_epochs=60, optimizer="adam", loss_function="categorical_crossentropy", num_conv_blocks=2, conv_filters_initial=32, dense_units=128, dropout_rate=0.5, random_seed=42, framework="tensorflow"
  - Log metrics per epoch: train_loss, train_accuracy, val_loss, val_accuracy (via epoch callback)
  - Log final metrics: test_loss, test_accuracy, test_precision, test_recall, test_f1
  - Save model artifact as "tensorflow_cnn_model"
  - Follows MLflow logging contract (contracts/mlflow-logging.md)

- [ ] T015 [P] [US1] Create pytorch/cnn/train.py: Training script with MLflow logging
  - Log identical parameters as TensorFlow script (names must match exactly)
  - Log metrics per epoch: train_loss, train_accuracy, val_loss, val_accuracy
  - Log final metrics: test_loss, test_accuracy, test_precision, test_recall, test_f1
  - Save model artifact as "pytorch_cnn_model"
  - Follows MLflow logging contract (contracts/mlflow-logging.md)

- [ ] T016 [P] [US1] Create tensorflow/cnn/evaluate.py: Evaluation utilities
  - Compute test metrics (accuracy, precision, recall, F1) using scikit-learn
  - Plot training history (train/val loss and accuracy over epochs)
  - Inference function for batch predictions

- [ ] T017 [P] [US1] Create pytorch/cnn/evaluate.py: Evaluation utilities
  - Compute test metrics (accuracy, precision, recall, F1) using scikit-learn
  - Plot training history (train/val loss and accuracy over epochs)
  - Inference function for batch predictions

- [ ] T018 [US1] Unit test in tests/test_tensorflow_cnn_models.py
  - Test model instantiation with seed=42
  - Test forward pass: input (1, 28, 28, 1) → output (1, 10)
  - Test softmax constraint: output sums to 1.0
  - Test accuracy on small subset ≥95%

- [ ] T019 [US1] Unit test in tests/test_pytorch_cnn_models.py
  - Test model instantiation with seed=42
  - Test forward pass: input (1, 1, 28, 28) → output (1, 10)
  - Test model.eval() and model.train() modes
  - Test accuracy on small subset ≥95%

- [ ] T020 [US1] Integration test: Run tensorflow/cnn/train.py for 2 epochs; verify MLflow logging
  - Verify experiment created: "cnn_mnist_tensorflow"
  - Verify all 11 parameters logged correctly (including random_seed=42)
  - Verify metrics logged per epoch (4 per epoch)
  - Verify final metrics logged (7 values)
  - Verify test_accuracy ≥ 0.98
  - Test in tests/integration/test_tensorflow_training.py

- [ ] T021 [US1] Integration test: Run pytorch/cnn/train.py for 2 epochs; verify MLflow logging
  - Same verification as T020 but for PyTorch
  - Verify experiment created: "cnn_mnist_pytorch"
  - Verify architecture equivalence to TensorFlow model
  - Test in tests/integration/test_pytorch_training.py

- [x] T022 [US1] Create README in tensorflow/cnn/README.md
  - How to train: `uv run python tensorflow/cnn/train.py --epochs 60`
  - How to view results: `mlflow ui --port 5000`
  - Explanation of architecture choices
  - Link to contracts/model-interface.md

- [x] T023 [US1] Create README in pytorch/cnn/README.md
  - How to train: `uv run python pytorch/cnn/train.py --epochs 60`
  - How to view results: `mlflow ui --port 5000`
  - Explanation of architecture choices
  - Link to contracts/model-interface.md

**Checkpoint**: User Story 1 complete - Both models train with identical MLflow logging; framework comparison enabled

---

## Phase 4: User Story 2 - CNN Learning Guides (Priority: P1)

**Goal**: Provide Jupyter notebooks as easy-to-follow guides demonstrating complete CNN pipeline for each framework (data → model → training → evaluation).

**Independent Test**: Both notebooks execute end-to-end without errors; 8+ markdown cells with explanations; visualizations present; training time <2 min per notebook; execution demonstrates full understanding of CNN and MLflow tracking.

### Implementation for User Story 2

- [ ] T024 [P] [US2] Create notebooks/tensorflow_cnn_guide.ipynb
  - 1. Setup & Imports: MLflow, TensorFlow, NumPy, matplotlib
  - 2. Load MNIST Data: Show shapes, visualize samples
  - 3. Data Preprocessing: Normalize to [0,1], create train/val/test split
  - 4. Model Definition: Explain each layer; show architecture
  - 5. Training Setup: MLflow experiment, parameters logging
  - 6. Training Loop: Execute with epoch-level metric logging
  - 7. Evaluation: Compute test metrics, plot training history
  - 8. Save Model: Log artifact to MLflow, retrieve run ID
  - Each section: 1-2 markdown cells explaining concepts + code cells
  - Per constitutional requirement Principle IV

- [ ] T025 [P] [US2] Create notebooks/pytorch_cnn_guide.ipynb
  - Identical conceptual structure to TensorFlow notebook (sections 1-8)
  - Same markdown explanations (parallel narrative)
  - PyTorch-specific implementations (DataLoader, forward pass, training loop)
  - 8+ markdown cells with explanations
  - Per constitutional requirement Principle IV

- [ ] T026 [US2] Create notebooks/README.md
  - How to run notebooks: `uv run jupyter notebook notebooks/`
  - Explanation of notebook structure
  - Why comparing both frameworks is valuable
  - Links to model documentation

- [ ] T027 [US2] Validate notebooks execute cleanly
  - Test in tests/integration/test_notebooks.py
  - Execute both notebooks programmatically
  - Verify: no errors, execution time <120 seconds each
  - Verify: at least 8 markdown cells per notebook

**Checkpoint**: User Story 2 complete - Notebooks serve as learning guides; reviewers can execute end-to-end

---

## Phase 5: User Story 3 - CNN Architecture Showcase (Priority: P1)

**Goal**: Demonstrate foundational CNN knowledge through well-documented model implementations with verified accuracy and validated architectural choices.

**Independent Test**: Models achieve ≥98% test accuracy; docstrings explain layer choices; architecture follows best practices for MNIST domain; code review confirms CNN competence.

### Implementation for User Story 3

- [ ] T028 [US3] Add comprehensive docstrings to tensorflow/cnn/models.py
  - Class docstring: Overall architecture summary, layer structure, hyperparameter explanation
  - Method docstrings: __init__, build (if Functional), forward
  - Parameter descriptions: kernel sizes, filter counts, rationale
  - Return descriptions: output shapes, value ranges

- [ ] T029 [US3] Add comprehensive docstrings to pytorch/cnn/models.py
  - Class docstring: Overall architecture summary, layer structure, hyperparameter explanation
  - Method docstrings: __init__, forward, with type hints
  - Parameter descriptions: input shapes, output shapes
  - Per Principle V

- [ ] T030 [P] [US3] Add docstrings to tensorflow/cnn/evaluate.py
  - Docstrings for all public functions: compute_metrics(), plot_history(), infer_batch()
  - Clear explanation of metric definitions
  - Per Principle V

- [ ] T031 [P] [US3] Add docstrings to pytorch/cnn/evaluate.py
  - Docstrings for all public functions
  - Clear explanation of metric definitions
  - Per Principle V

- [ ] T032 [US3] Verify model accuracy ≥98% in production runs
  - Train both models for full 60 epochs
  - Document final accuracy in tensorflow/cnn/README.md
  - Document final accuracy in pytorch/cnn/README.md
  - Include test loss and other metrics

- [ ] T033 [US3] Update main project README.md
  - Add section: "CNN Models" with links to tensorflow/cnn/ and pytorch/cnn/
  - Explain CNN architecture choices (2 conv blocks, pooling, why effective for MNIST)
  - Point to quickstart.md for validation guide
  - Per Principle VI

**Checkpoint**: User Story 3 complete - CNN implementations are well-documented, high-accuracy, and showcase architectural knowledge

---

## Phase 6: User Story 4 - Reproducible Experiments (Priority: P2)

**Goal**: Ensure experiments are fully reproducible through fixed seeds, proper serialization, and artifact management via MLflow.

**Independent Test**: Training runs are identical when using seed=42; model artifacts loadable from MLflow; loaded models produce identical predictions; experiment parameters sufficient for replication without additional documentation.

### Implementation for User Story 4

- [ ] T034 [P] [US4] Implement seed=42 enforcement in tensorflow/cnn/models.py
  - Set tf.random.set_seed(seed) in __init__
  - Document seed requirement in docstring
  - Add test: verify weight initialization is deterministic

- [ ] T035 [P] [US4] Implement seed=42 enforcement in pytorch/cnn/models.py
  - Set torch.manual_seed(seed) in __init__
  - Document seed requirement in docstring
  - Add test: verify weight initialization is deterministic

- [ ] T036 [US4] Add model serialization test in tests/test_model_serialization.py
  - Test TensorFlow model save/load via MLflow
  - Test PyTorch model save/load via MLflow
  - Verify loaded models produce identical predictions on test set
  - Verify accuracy preserved

- [ ] T037 [US4] Add reproducibility test in tests/test_reproducibility.py
  - Train model twice with seed=42, same hyperparameters
  - Verify weight initialization is identical
  - Verify test accuracy is identical
  - Test for both frameworks

- [ ] T038 [US4] Document reproducibility in tensorflow/cnn/README.md
  - How to load trained model: `mlflow.tensorflow.load_model(...)`
  - How to use model for inference: example code
  - Seed requirement: seed=42 must be used

- [ ] T039 [US4] Document reproducibility in pytorch/cnn/README.md
  - How to load trained model: `mlflow.pytorch.load_model(...)`
  - How to use model for inference: example code
  - Seed requirement: seed=42 must be used

**Checkpoint**: User Story 4 complete - Experiments are fully reproducible; models can be loaded and reused exactly

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Code quality, testing, documentation, and validation across all user stories

- [ ] T040 [P] Linting: tensorflow/cnn/ with flake8
  - Command: `uv run flake8 tensorflow/cnn/ --max-line-length=100`
  - Target: Zero warnings

- [ ] T041 [P] Linting: pytorch/cnn/ with flake8
  - Command: `uv run flake8 pytorch/cnn/ --max-line-length=100`
  - Target: Zero warnings

- [ ] T042 [P] Linting: utils/ with flake8
  - Command: `uv run flake8 utils/ --max-line-length=100`
  - Target: Zero warnings

- [ ] T043 [P] Pylint checks: tensorflow/cnn/
  - Command: `uv run pylint tensorflow/cnn/ --disable=C0111 --disable=R0913`
  - Target: Zero warnings

- [ ] T044 [P] Pylint checks: pytorch/cnn/
  - Command: `uv run pylint pytorch/cnn/ --disable=C0111 --disable=R0913`
  - Target: Zero warnings

- [ ] T045 [P] Pylint checks: utils/
  - Command: `uv run pylint utils/ --disable=C0111 --disable=R0913`
  - Target: Zero warnings

- [ ] T046 Run all unit tests
  - Command: `uv run pytest tests/test_*.py -v`
  - Target: All pass, coverage ≥70%

- [ ] T047 Run all integration tests
  - Command: `uv run pytest tests/integration/ -v`
  - Target: All pass

- [ ] T048 Execute quickstart.md validation guide
  - Run through all 10 validation steps from quickstart.md
  - Verify: Code quality, Model interfaces, Unit tests, Training + MLflow logging, Framework comparison, Notebooks, Reproducibility, Documentation
  - Document results

- [ ] T049 Create/update IMPLEMENTATION_NOTES.md
  - Summary of what was implemented
  - Architecture decisions and rationale
  - Known limitations or future improvements
  - Instructions for extending (adding more model variants, etc.)

- [ ] T050 Final sanity check: Run complete training pipeline
  - Train TensorFlow CNN for 60 epochs
  - Train PyTorch CNN for 60 epochs
  - Verify both achieve ≥98% accuracy
  - Verify MLflow experiments created and comparable
  - Verify notebooks execute without errors
  - Total training time <10 minutes (both models)

**Checkpoint**: All requirements met; code polished; tests passing; documentation complete; ready for portfolio presentation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - **BLOCKS all user stories**
- **User Stories (Phases 3-6)**: All depend on Foundational phase completion
  - US1 (P1): Can start immediately after Foundational
  - US2 (P1): Can start immediately after Foundational (independent of US1)
  - US3 (P1): Can start immediately after Foundational (enhances US1 models)
  - US4 (P2): Can start after US1 or in parallel (enhances reproducibility)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### Within Each User Story

- Implementation tasks before tests (though tests should fail first)
- Models before training scripts
- Training before evaluation
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1**: All tasks sequential but quick (directory/file creation)
- **Phase 2**: T007, T008 (data loaders) can run in parallel; T010 conftest can run in parallel
- **Phase 3 (US1)**:
  - T012 and T013 (models) in parallel
  - T014 and T015 (train scripts) in parallel (after models)
  - T016 and T017 (evaluate functions) in parallel
  - T018 and T019 (model unit tests) in parallel
- **Phase 4 (US2)**:
  - T024 and T025 (notebooks) in parallel
- **Phase 5 (US3)**:
  - T028 and T029 (model docstrings) in parallel
  - T030 and T031 (evaluate docstrings) in parallel
- **Phase 6 (US4)**:
  - T034 and T035 (seed enforcement) in parallel
- **Phase 7 (Polish)**:
  - All flake8/pylint tasks (T040-T045) in parallel
  - Tests can run in parallel

---

## Parallel Example: Phase 3 (US1) Rapid Implementation

### Option 1: Single Developer (Sequential)

```
1. T012: Create TF model
2. T013: Create PT model (after T012 - understand architecture once)
3. T014-T015: Create train scripts (after models)
4. T016-T017: Create evaluate functions
5. T018-T021: Write and run tests
```

### Option 2: Multiple Developers (Parallel)

```
Developer A: T012 (TF model) + T014 (TF train) + T016 (TF evaluate)
Developer B: T013 (PT model) + T015 (PT train) + T017 (PT evaluate)
(Both can work in parallel once they understand the architecture)

Then parallel:
Dev A: T018-T020 (TF tests)
Dev B: T019-T021 (PT tests)
Dev C: T022-T023 (READMEs)
```

### Option 3: Rapid MVP (Minimum Parallelism)

```
1. T012, T013: Models (pair programming or sequential with design review)
2. T014, T015: Training scripts (parallel - different files, same utils)
3. Tests: T018-T021 (parallel - different files)
4. READMEs: T022-T023 (parallel - different files)
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Complete Phase 1: Setup (~30 min)
2. Complete Phase 2: Foundational (~2-3 hours)
3. Complete Phase 3: User Story 1 (~4-5 hours with testing)
4. Complete Phase 4: User Story 2 (~2-3 hours with notebooks)
5. **STOP and VALIDATE**: Run quickstart.md; verify both models train with MLflow, notebooks execute
6. Deploy/demo as portfolio piece

**Estimated Time**: ~10-14 hours for MVP

### Incremental Delivery (All User Stories)

1. Complete MVP as above
2. Add Phase 5: User Story 3 (~1-2 hours - documentation, validation)
3. Add Phase 6: User Story 4 (~2-3 hours - reproducibility testing)
4. Complete Phase 7: Polish & Testing (~2-3 hours - linting, final validation)

**Estimated Time**: ~16-22 hours total for complete feature

### Single Developer Timeline

- **Day 1 (4-5 hours)**: Setup + Foundational phases
- **Day 2 (4-5 hours)**: User Story 1 (models + training)
- **Day 3 (2-3 hours)**: User Story 2 (notebooks)
- **Day 4 (1-2 hours)**: User Stories 3 + 4 + Polish

---

## Task Completion Notes

- Mark [x] when a task completes
- After each phase checkpoint, validate independently before proceeding
- Commit after each task or logical group (e.g., after T012, after T014-T015)
- Stop at any checkpoint to demo/validate
- Use `uv run` prefix for all command execution (environment isolation)
- Refer to contracts/ and data-model.md for specifications during implementation

---

## Phase 8: Convergence (Gap Closure)

**Purpose**: Close gaps identified between specification/plan/tasks and current codebase implementation. All remaining work to satisfy spec requirements, user stories, and constitutional principles.

**CRITICAL**: Principle IV (Jupyter Notebooks) is unfulfilled; notebooks are primary learning artifacts and constitutional requirement.

### Data Loading & Infrastructure

- [x] T051 Create tensorflow/cnn/data.py with MNIST data loading and preprocessing (per T007)
  - Implement load_mnist() function
  - Handle automatic download if needed
  - Return normalized (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
  - Input shape verification: (N, 28, 28, 1) for TensorFlow
  - Normalize to [0, 1] range
  - Test split: 50k train, 10k val, 10k test

- [x] T052 Create pytorch/cnn/data.py with MNIST data loading and DataLoader creation (per T008)
  - Implement load_mnist() function returning PyTorch DataLoaders
  - Verify input shape: (N, 1, 28, 28) for PyTorch (NCHW format)
  - Use torch.utils.data.DataLoader with batch_size=32
  - Normalize to [0, 1] range
  - Handle train/val/test splits consistently with TensorFlow

### Model Implementations (User Story 1)

- [x] T053 [P] Create tensorflow/cnn/models.py: MNISTCNNModel class (per T012)
  - Inherit from keras.Sequential
  - Constructor params: input_shape=(28,28,1), num_classes=10, num_conv_blocks=2, conv_filters_initial=32, dense_units=128, dropout_rate=0.5, seed=42
  - Implement 2 Conv→MaxPool blocks, Flatten, Dense(128, ReLU), Dropout, Dense(10, Softmax)
  - Set tf.random.set_seed(seed) for reproducibility
  - Compile with Adam optimizer, categorical_crossentropy loss, accuracy metric
  - Include complete docstring explaining architecture and hyperparameters
  - Verify output shape: (batch_size, 10) with softmax probabilities

- [x] T054 [P] Create pytorch/cnn/models.py: MNISTCNNModel class (per T013)
  - Inherit from torch.nn.Module
  - Constructor params: input_channels=1, num_classes=10, num_conv_blocks=2, conv_filters_initial=32, dense_units=128, dropout_rate=0.5, seed=42
  - Implement forward() method matching architecture from contracts/model-interface.md
  - Set torch.manual_seed(seed) for reproducibility
  - Include train()/eval() mode support for dropout
  - Include complete docstring with type hints and architecture explanation
  - Verify output shape: (batch_size, 10) with logits (no softmax in model)

### Training Scripts with MLflow Integration

- [x] T055 Create tensorflow/cnn/train.py: Training script with MLflow logging (per T014)
  - Parse CLI args or use defaults: epochs=60, batch_size=32, learning_rate=0.001
  - Use utils/mlflow_config.py to setup MLflow experiment "cnn_mnist_tensorflow"
  - Log all 11 parameters per MLflow contract: learning_rate, batch_size, num_epochs, optimizer="adam", loss_function="categorical_crossentropy", num_conv_blocks=2, conv_filters_initial=32, dense_units=128, dropout_rate=0.5, random_seed=42, framework="tensorflow"
  - Log train/val metrics per epoch using MLflow (step=epoch)
  - Compute test metrics (accuracy, precision, recall, F1) using utils/metrics.py
  - Log final metrics: test_loss, test_accuracy, test_precision, test_recall, test_f1, training_time_seconds, inference_time_ms_per_batch
  - Save model artifact to MLflow as "tensorflow_cnn_model"
  - Achieve ≥98% test accuracy per SC-001

- [x] T056 Create pytorch/cnn/train.py: Training script with MLflow logging (per T015)
  - Parse CLI args or use defaults: epochs=60, batch_size=32, learning_rate=0.001
  - Use utils/mlflow_config.py to setup MLflow experiment "cnn_mnist_pytorch"
  - Log identical 11 parameters to TensorFlow for fair comparison
  - Implement training loop with optimizer step, loss computation, backward pass
  - Log train/val metrics per epoch (step=epoch)
  - Compute test metrics using utils/metrics.py
  - Log final metrics: test_loss, test_accuracy, test_precision, test_recall, test_f1, training_time_seconds, inference_time_ms_per_batch
  - Save model artifact to MLflow as "pytorch_cnn_model"
  - Achieve ≥98% test accuracy per SC-001

### Evaluation Utilities

- [x] T057 [P] Create tensorflow/cnn/evaluate.py: Evaluation utilities (per T016)
  - Implement compute_metrics(y_true, y_pred, y_probs) → (loss, accuracy, precision, recall, f1)
  - Implement plot_history(history_dict) → saves training_history.png
  - Implement infer_batch(model, batch_data) → predictions with batch handling
  - Import from utils/metrics.py for consistency
  - Include docstrings for all public functions

- [x] T058 [P] Create pytorch/cnn/evaluate.py: Evaluation utilities (per T017)
  - Implement compute_metrics(y_true, y_pred, y_logits) → (loss, accuracy, precision, recall, f1)
  - Implement plot_history(history_dict) → saves training_history.png
  - Implement infer_batch(model, batch_data) → predictions with batch handling
  - Use torch evaluation patterns (model.eval(), torch.no_grad())
  - Include docstrings for all public functions

### Testing Infrastructure

- [ ] T059 Create tests/conftest.py with pytest fixtures (per T010)
  - Provide fixture for temporary MNIST dataset
  - Provide fixture for MLflow tracking setup/cleanup
  - Provide fixture with seed=42 enforcement for reproducibility

- [ ] T060 Create tests/test_data_loading.py: Verify data loading consistency (per T011)
  - Test tensorflow/cnn/data.py: verify shapes, normalization, split counts
  - Test pytorch/cnn/data.py: verify shapes, DataLoader batch sizes
  - Test both frameworks load identical data splits (sample comparison)
  - Target: All data tests pass

### Model Unit Tests

- [ ] T061 Create tests/test_tensorflow_cnn_models.py (per T018)
  - Test model instantiation with seed=42
  - Test forward pass: input (1, 28, 28, 1) → output (1, 10)
  - Test softmax constraint: output sums to 1.0 ± tolerance
  - Test model.compile() includes optimizer and loss
  - Test accuracy on small subset ≥95%
  - Target: All model tests pass

- [ ] T062 Create tests/test_pytorch_cnn_models.py (per T019)
  - Test model instantiation with seed=42
  - Test forward pass: input (1, 1, 28, 28) → output (1, 10)
  - Test model.train()/model.eval() modes
  - Test model.parameters() access for optimizer
  - Test accuracy on small subset ≥95%
  - Target: All model tests pass

### Integration Tests

- [ ] T063 Create tests/integration/test_tensorflow_training.py (per T020)
  - Run tensorflow/cnn/train.py for 2 epochs
  - Verify experiment created: "cnn_mnist_tensorflow"
  - Verify all 11 parameters logged (including random_seed=42)
  - Verify metrics logged per epoch (4 metrics × epochs)
  - Verify final metrics logged (7 values)
  - Verify test_accuracy ≥ 0.98
  - Target: Integration test passes

- [ ] T064 Create tests/integration/test_pytorch_training.py (per T021)
  - Run pytorch/cnn/train.py for 2 epochs
  - Verify experiment created: "cnn_mnist_pytorch"
  - Verify all 11 parameters logged (including random_seed=42)
  - Verify metrics logged per epoch (4 metrics × epochs)
  - Verify architecture equivalence to TensorFlow model
  - Verify final metrics logged (7 values)
  - Verify test_accuracy ≥ 0.98
  - Target: Integration test passes

### Documentation & Learning Guides (User Story 2 - CRITICAL)

- [x] T065 [P] Create notebooks/tensorflow_cnn_guide.ipynb (per T024, Principle IV)
  - Section 1: Setup & Imports (MLflow, TensorFlow, NumPy, matplotlib, sklearn)
  - Section 2: Load MNIST Data (show shapes, visualize 10 sample images)
  - Section 3: Data Preprocessing (normalization [0,1], train/val/test split)
  - Section 4: Model Definition (explain Conv, MaxPool, Dense, Dropout layers; show summary)
  - Section 5: Training Setup (MLflow experiment, parameter logging explanation)
  - Section 6: Training Loop (execute with epoch-level metric logging; 5-10 epochs for quick execution)
  - Section 7: Evaluation (compute test metrics, plot training_history, show confusion matrix)
  - Section 8: Save Model (log artifact to MLflow, retrieve run ID for reference)
  - Add 1-2 markdown cells per section explaining concepts
  - Total ≥8 markdown cells minimum (per SC-004)
  - Execution time <2 minutes on CPU (per SC-003)
  - Per Principle IV: must serve as learning guide with clear explanations

- [x] T066 [P] Create notebooks/pytorch_cnn_guide.ipynb (per T025, Principle IV)
  - Identical conceptual structure to TensorFlow notebook (8 sections)
  - Same markdown explanations and visualizations (parallel narrative)
  - PyTorch-specific implementations: DataLoader, forward pass, training loop with optimizer
  - Add 1-2 markdown cells per section explaining concepts
  - Total ≥8 markdown cells minimum (per SC-004)
  - Execution time <2 minutes on CPU (per SC-003)
  - Demonstrate framework differences clearly while maintaining equivalent pipeline
  - Per Principle IV: must serve as learning guide with clear explanations

- [x] T067 Create notebooks/README.md (per T026)
  - Explain how to run notebooks: `uv run jupyter notebook notebooks/`
  - Describe notebook structure and learning objectives
  - Explain why comparing both frameworks is valuable
  - Link to tensorflow/cnn/README.md and pytorch/cnn/README.md for framework-specific info
  - Point to contracts/model-interface.md for architectural details

- [ ] T068 Create tests/integration/test_notebooks.py (per T027)
  - Execute both notebooks programmatically using nbconvert or nbformat
  - Verify: no errors during execution
  - Verify: execution time <120 seconds each
  - Verify: at least 8 markdown cells per notebook
  - Target: Both notebooks execute cleanly

### Architecture Documentation (User Story 3)

- [ ] T069 Add comprehensive docstrings to tensorflow/cnn/models.py (per T028)
  - Class docstring: CNN architecture summary, why 2 blocks + 2 dense layers
  - Explain conv filter progression (32 → 64) and why effective for MNIST
  - Document all hyperparameters and their roles
  - Method docstrings: __init__, build (if Functional), forward pass behavior
  - Reference contracts/model-interface.md for validation details

- [ ] T070 Add comprehensive docstrings to pytorch/cnn/models.py (per T029)
  - Class docstring: CNN architecture summary, equivalence to TensorFlow
  - Explain Conv2d → MaxPool progression and design rationale
  - Document all hyperparameters and their roles
  - Method docstrings: __init__, forward, train/eval modes
  - Include type hints on public methods
  - Per Principle V: complete docstrings per spec

- [ ] T071 [P] Add docstrings to tensorflow/cnn/evaluate.py (per T030)
  - Docstrings for all public functions: compute_metrics(), plot_history(), infer_batch()
  - Explain metric definitions (accuracy, precision, recall, F1)
  - Include parameter and return types
  - Per Principle V: complete documentation

- [ ] T072 [P] Add docstrings to pytorch/cnn/evaluate.py (per T031)
  - Docstrings for all public functions: compute_metrics(), plot_history(), infer_batch()
  - Explain metric definitions and PyTorch-specific patterns
  - Include parameter and return types
  - Per Principle V: complete documentation

- [ ] T073 Run full production training for accuracy validation (per T032)
  - Train TensorFlow CNN for full 60 epochs, document final accuracy in tensorflow/cnn/README.md
  - Train PyTorch CNN for full 60 epochs, document final accuracy in pytorch/cnn/README.md
  - Include test_loss, test_precision, test_recall, test_f1 in READMEs
  - Verify both achieve ≥98% accuracy (per SC-001)
  - Verify MLflow experiments display comparable results

- [ ] T074 Update main project README.md (per T033, Principle VI)
  - Add section: "CNN Models"
  - Explain CNN architecture choices: why 2 conv blocks, why MaxPooling, why effective for MNIST
  - Link to tensorflow/cnn/README.md and pytorch/cnn/README.md
  - Link to quickstart.md for validation guide
  - Explain framework comparison value
  - Per Principle VI: comprehensive documentation

### Reproducibility & Serialization (User Story 4)

- [ ] T075 [P] Implement seed=42 enforcement in tensorflow/cnn/models.py (per T034)
  - Call tf.random.set_seed(seed) in __init__ with provided seed parameter
  - Document seed requirement in docstring: "seed=42 required for reproducibility"
  - Add test to verify weight initialization is deterministic across runs

- [ ] T076 [P] Implement seed=42 enforcement in pytorch/cnn/models.py (per T035)
  - Call torch.manual_seed(seed) in __init__ with provided seed parameter
  - Document seed requirement in docstring: "seed=42 required for reproducibility"
  - Add test to verify weight initialization is deterministic across runs

- [ ] T077 Create tests/test_model_serialization.py (per T036)
  - Test TensorFlow model save/load via MLflow: train → save → load → predict
  - Test PyTorch model save/load via MLflow: train → save → load → predict
  - Verify loaded models produce identical predictions on test set (within floating-point tolerance)
  - Verify accuracy preserved after load

- [ ] T078 Create tests/test_reproducibility.py (per T037)
  - Train both models twice with seed=42, identical hyperparameters
  - Verify weight initialization is identical between runs
  - Verify test accuracy is identical between runs
  - Test both TensorFlow and PyTorch

- [ ] T079 Update tensorflow/cnn/README.md with reproducibility guide (per T038)
  - How to load trained model: example code with mlflow.tensorflow.load_model()
  - How to use model for inference: example batch prediction code
  - Seed requirement: "seed=42 MUST be used for reproducibility"
  - Link to MLflow tracking for parameter verification

- [ ] T080 Update pytorch/cnn/README.md with reproducibility guide (per T039)
  - How to load trained model: example code with mlflow.pytorch.load_model()
  - How to use model for inference: example batch prediction code with torch.no_grad()
  - Seed requirement: "seed=42 MUST be used for reproducibility"
  - Link to MLflow tracking for parameter verification

### Code Quality & Validation (Phase 7)

- [ ] T081 [P] Linting: tensorflow/cnn/ with flake8 (per T040)
  - Command: `uv run flake8 tensorflow/cnn/ --max-line-length=100`
  - Target: Zero warnings
  - Verify all files: __init__.py, models.py, data.py, train.py, evaluate.py

- [ ] T082 [P] Linting: pytorch/cnn/ with flake8 (per T041)
  - Command: `uv run flake8 pytorch/cnn/ --max-line-length=100`
  - Target: Zero warnings
  - Verify all files: __init__.py, models.py, data.py, train.py, evaluate.py

- [ ] T083 [P] Linting: utils/ with flake8 (per T042)
  - Command: `uv run flake8 utils/ --max-line-length=100`
  - Target: Zero warnings

- [ ] T084 [P] Pylint checks: tensorflow/cnn/ (per T043)
  - Command: `uv run pylint tensorflow/cnn/ --disable=C0111 --disable=R0913`
  - Target: Zero warnings

- [ ] T085 [P] Pylint checks: pytorch/cnn/ (per T044)
  - Command: `uv run pylint pytorch/cnn/ --disable=C0111 --disable=R0913`
  - Target: Zero warnings

- [ ] T086 [P] Pylint checks: utils/ (per T045)
  - Command: `uv run pylint utils/ --disable=C0111 --disable=R0913`
  - Target: Zero warnings

- [ ] T087 Run all unit tests (per T046)
  - Command: `uv run pytest tests/test_*.py -v`
  - Target: All pass, coverage ≥70%

- [ ] T088 Run all integration tests (per T047)
  - Command: `uv run pytest tests/integration/ -v`
  - Target: All pass

- [ ] T089 Execute quickstart.md validation guide (per T048)
  - Run through all validation steps from quickstart.md
  - Verify: Code quality (linting zero warnings), Model interfaces (contracts satisfied), Unit tests (all pass), Training + MLflow (experiment created, parameters logged), Framework comparison (both models tracked), Notebooks (both execute <2 min), Reproducibility (seed=42 enforced), Documentation (READMEs complete)
  - Document results

- [ ] T090 Create/update IMPLEMENTATION_NOTES.md (per T049)
  - Summary of CNN implementation
  - Architecture decisions: why 2 conv blocks, filter progression, layer choices
  - Rationale for hyperparameters
  - Known limitations or future improvements
  - Instructions for extending (adding more architectures, adjusting hyperparameters, etc.)

- [ ] T091 Final sanity check: Complete training pipeline (per T050)
  - Train TensorFlow CNN for 60 epochs, verify ≥98% accuracy
  - Train PyTorch CNN for 60 epochs, verify ≥98% accuracy
  - Verify both MLflow experiments created with all 11 parameters logged
  - Verify metrics comparable across frameworks (same names, same value ranges)
  - Execute both notebooks end-to-end, verify <2 min execution each
  - Total training time <10 minutes (both models)
  - **SUCCESS CRITERION**: Both models ≥98%, both notebooks <2 min, MLflow experiments comparable

---

## Validation Checklist

- [ ] All tasks have checkbox, ID, optional [P] marker, optional [Story] label, clear description with file path
- [ ] Setup phase creates all directories
- [ ] Foundational phase creates all utilities before user stories
- [ ] Each user story is independently completable and testable
- [ ] Tests included (per specification)
- [ ] README files provide clear usage instructions
- [ ] Linting targets: 0 warnings
- [ ] quickstart.md runs successfully end-to-end
- [ ] Both models achieve ≥98% accuracy
- [ ] MLflow experiments visible and comparable
- [ ] Notebooks execute <2 min each
- [ ] Code follows Principle V (documentation, docstrings)
