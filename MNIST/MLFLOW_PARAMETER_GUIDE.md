# MLflow Parameters & Metrics Visual Guide

## Experiment Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  MLflow Experiment (e.g., "mnist_tensorflow_fcnn")              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  MLflow Run (Single Training Session)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────┬──────────────────────────┐
    ↓                         ↓                          ↓
    
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  PARAMETERS      │  │  PER-EPOCH       │  │  FINAL METRICS   │
│  (11 required)   │  │  METRICS (4)     │  │  (7 required)    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
      ↓ (pre-training)     ↓ (during)         ↓ (post-training)
      
Logged once            Logged at:            Logged once
before training        step=0,1,2,...        after training
```

## Parameter Quick Reference

### Group 1: Optimization Parameters

```
┌─ OPTIMIZATION ─────────────────────┐
│                                    │
│ learning_rate    : 0.001 (float)  │
│ batch_size       : 32-64  (int)   │
│ num_epochs       : 60     (int)   │
│ optimizer        : "adam" (str)   │
│ loss_function    : "categorical"  │
│                   "crossentropy"  │
│                      (str)         │
│ random_seed      : 42   (int)     │ ← MUST BE 42
│                                    │
└────────────────────────────────────┘
```

### Group 2: Architecture Parameters

```
┌─ ARCHITECTURE ─────────────────────┐
│                                    │
│ FCNN:                              │
│   num_conv_blocks      : 0 (int)  │
│   conv_filters_initial : 0 (int)  │
│   dense_units          : 128 (int)│
│   dropout_rate         : 0.2 (flt)│
│                                    │
│ CNN:                               │
│   num_conv_blocks      : 2 (int)  │
│   conv_filters_initial : 32 (int) │
│   dense_units          : 128 (int)│
│   dropout_rate         : 0.5 (flt)│
│                                    │
└────────────────────────────────────┘
```

### Group 3: Context Parameters

```
┌─ CONTEXT ──────────────────────────┐
│                                    │
│ framework : "tensorflow"  (str)    │
│            or "pytorch"            │
│                                    │
└────────────────────────────────────┘
```

## Metrics Quick Reference

### Per-Epoch Metrics (With step=epoch)

```
Logged at each epoch:

step=0:
  train_loss       = 0.4
  train_accuracy   = 0.91
  val_loss         = 0.35
  val_accuracy     = 0.93

step=1:
  train_loss       = 0.25
  train_accuracy   = 0.94
  val_loss         = 0.22
  val_accuracy     = 0.95

step=2:
  train_loss       = 0.15
  train_accuracy   = 0.96
  val_loss         = 0.13
  val_accuracy     = 0.96

... (continues for all epochs)
```

### Final Metrics (No step parameter)

```
Logged once after training:

test_loss                    = 0.10
test_accuracy                = 0.985
test_precision               = 0.984  (macro-averaged)
test_recall                  = 0.985  (macro-averaged)
test_f1                      = 0.984  (macro-averaged)
training_time_seconds        = 65.4
inference_time_ms_per_batch  = 2.3
```

## Data Flow: TensorFlow Example

```
Input                 →    Log              →    MLflow Storage
────────────────────────────────────────────────────────────────

TRAINING CODE                          MLFLOW CONFIG CALL
─────────────────                      ──────────────────

learning_rate = 0.001    →  mlflow_cfg.log_params({
batch_size = 32             "learning_rate": 0.001,
num_epochs = 60             "batch_size": 32,
optimizer = "adam"          "num_epochs": 60,
loss_fn = "categorical      ...
          crossentropy"     })
num_conv_blocks = 0                    ↓
conv_filters = 0                    VALIDATES
dense_units = 128                   (checks all 11 present)
dropout = 0.2                           ↓
seed = 42                           STORES IN MLflow
framework = "tensorflow"            {"learning_rate": "0.001",
                                     "batch_size": "32",
TRAINING LOOP:                       ...}

for epoch in range(60):
  train_loss = 0.15
  train_acc = 0.96       →  mlflow_cfg.log_epoch_metrics(
  val_loss = 0.12           epoch=epoch,
  val_acc = 0.97            train_loss=train_loss,
                            train_accuracy=train_acc,
                        →   val_loss=val_loss,
                            val_accuracy=val_acc)
                                    ↓
                                STORES
                        {"train_loss": 0.15, step: 0}
                        {"train_loss": 0.12, step: 1}
                        ... (for all epochs)

EVALUATION:
y_pred = model.predict(x_test)
y_true = ...
precision, recall, f1 = calculate_metrics(y_true, y_pred)

test_loss = 0.10       →  mlflow_cfg.log_final_metrics(
test_acc = 0.985           test_loss=test_loss,
test_prec = 0.984          test_accuracy=test_acc,
test_recall = 0.985    →   test_precision=test_prec,
test_f1 = 0.984            test_recall=test_recall,
train_time = 65.4          test_f1=test_f1,
infer_time = 2.3           inference_time_ms_per_batch=...)
                                    ↓
                                STORES
                        {"test_loss": "0.10"}
                        {"test_accuracy": "0.985"}
                        ... (7 metrics total)
```

## Validation Flow

```
┌─ Training Script Starts ─────────────────────────────┐
│                                                      │
│  mlflow_cfg = MLflowConfig("mnist_tensorflow_fcnn")│
│                                                      │
└──────────────────────┬───────────────────────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  mlflow_cfg.start_run()          │
        │  (initializes new run)           │
        └──────────────────┬───────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │  mlflow_cfg.log_params({...})    │
        │                                  │
        │  ✓ Validates 11 params present   │
        │  ✓ Normalizes parameter names    │
        │  ✓ Converts values to strings    │
        │  ✓ Stores in MLflow              │
        │                                  │
        │  ✗ Fails if params missing       │
        └──────────────────┬───────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │  TRAINING LOOP                   │
        │                                  │
        │  for epoch in range(num_epochs): │
        │    mlflow_cfg.log_epoch_metrics()│
        │    ✓ Logs 4 metrics at step=ep   │
        └──────────────────┬───────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │  EVALUATION                      │
        │                                  │
        │  mlflow_cfg.log_final_metrics()  │
        │  ✓ Logs 7 final metrics          │
        │  ✓ No step parameter (logged 1x) │
        │  ✓ Calculates training_time_sec  │
        └──────────────────┬───────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │  mlflow_cfg.end_run()            │
        │  (closes run)                    │
        └──────────────────┬───────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │  MLflow Run Complete             │
        │                                  │
        │  Total logged:                   │
        │  • 11 parameters                 │
        │  • 4 × num_epochs per-epoch met  │
        │  • 7 final metrics               │
        └──────────────────────────────────┘
```

## Comparison: Framework Equivalence

### TensorFlow FCNN
```
Parameters logged:
  learning_rate=0.001, batch_size=32, num_epochs=60, 
  optimizer="adam", loss_function="categorical_crossentropy",
  num_conv_blocks=0, conv_filters_initial=0,
  dense_units=128, dropout_rate=0.2, random_seed=42,
  framework="tensorflow"
  ✓ 11/11 parameters

Metrics:
  Per-epoch: train_loss, train_accuracy, val_loss, val_accuracy
  ✓ 4/4 per-epoch metrics × 60 epochs = 240 metric entries
  
  Final: test_loss, test_accuracy, test_precision, test_recall,
         test_f1, training_time_seconds, inference_time_ms_per_batch
  ✓ 7/7 final metrics

Result: ✓ STANDARD, can compare with PyTorch
```

### PyTorch FCNN
```
Parameters logged:
  learning_rate=0.001, batch_size=64, num_epochs=60, 
  optimizer="adam", loss_function="crossentropyloss",
  num_conv_blocks=0, conv_filters_initial=0,
  dense_units=128, dropout_rate=0.2, random_seed=42,
  framework="pytorch"
  ✓ 11/11 parameters (SAME NAMES as TensorFlow)

Metrics:
  Per-epoch: train_loss, train_accuracy, val_loss, val_accuracy
  ✓ 4/4 per-epoch metrics × 60 epochs = 240 metric entries
  
  Final: test_loss, test_accuracy, test_precision, test_recall,
         test_f1, training_time_seconds, inference_time_ms_per_batch
  ✓ 7/7 final metrics (SAME NAMES as TensorFlow)

Result: ✓ STANDARD, EQUIVALENT TO TENSORFLOW → Fair comparison!
```

## Common Issues & Fixes

### Issue 1: Missing Parameter in log_params()

```
Code:
  mlflow_cfg.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    # ... only 6 params, missing 5!
  })

Error:
  ValueError: Missing required parameters: 
  {'num_epochs', 'num_conv_blocks', ...}

Fix:
  Ensure all 11 parameters are present in the dict
  (use the template from MLFLOW_QUICK_REFERENCE.md)
```

### Issue 2: Wrong Inference Time Metric Name

```
Before (❌):
  mlflow.log_metric("inference_time_ms", 2.3)

After (✓):
  mlflow_cfg.log_final_metrics(
    ...,
    inference_time_ms_per_batch=2.3
  )

Note: MLflowConfig handles naming automatically
```

### Issue 3: Missing test_f1 Metric

```
Before (❌):
  mlflow.log_metric("test_loss", 0.1)
  mlflow.log_metric("test_accuracy", 0.98)
  # ... no test_f1

After (✓):
  precision, recall, f1 = calculate_metrics(y_true, y_pred)
  mlflow_cfg.log_final_metrics(
    ...,
    test_f1=f1  # now included
  )
```

### Issue 4: Forgetting to Log framework Parameter

```
Before (❌):
  mlflow_cfg.log_params({
    "learning_rate": 0.001,
    # ... no framework
  })

Error:
  ValueError: Missing required parameters: {'framework', ...}

After (✓):
  mlflow_cfg.log_params({
    "learning_rate": 0.001,
    ...,
    "framework": "tensorflow"  # must include
  })
```

## Step-by-Step Template

Use this template for any new FCNN/CNN training script:

```python
from utils.mlflow_config import MLflowConfig
from utils.metrics import calculate_metrics

# 1. Initialize MLflow
mlflow_cfg = MLflowConfig("experiment_name_here", "sqlite:///./mlflow.db")

# 2. Start run
mlflow_cfg.start_run(tags={
    "framework": "tensorflow",  # or "pytorch"
    "architecture": "fcnn",     # or "cnn"
})

try:
    # 3. Log parameters (BEFORE training)
    mlflow_cfg.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 60,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "num_conv_blocks": 0,           # or 2 for CNN
        "conv_filters_initial": 0,      # or 32 for CNN
        "dense_units": 128,
        "dropout_rate": 0.2,
        "random_seed": 42,
        "framework": "tensorflow",
    })
    
    # 4. Training loop - log per-epoch metrics
    for epoch in range(num_epochs):
        # ... training code ...
        mlflow_cfg.log_epoch_metrics(
            epoch=epoch,
            train_loss=computed_train_loss,
            train_accuracy=computed_train_acc,
            val_loss=computed_val_loss,
            val_accuracy=computed_val_acc,
        )
    
    # 5. Evaluate - calculate metrics
    y_pred = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)  # or your method
    y_pred_class = np.argmax(y_pred, axis=1)
    
    precision, recall, f1 = calculate_metrics(y_true, y_pred_class)
    
    # 6. Log final metrics
    mlflow_cfg.log_final_metrics(
        test_loss=computed_test_loss,
        test_accuracy=computed_test_acc,
        test_precision=precision,
        test_recall=recall,
        test_f1=f1,
        inference_time_ms_per_batch=computed_infer_time,
    )
    
finally:
    # 7. Always close the run
    mlflow_cfg.end_run()
```

---

**Print this guide and keep it handy when implementing CNN models!**
