# Parameter Tuning Guide: Fusion Cuisine Learning System

This guide provides specific recommendations for tuning parameters based on different training objectives and constraints.

## Table of Contents

1. [Training Objectives](#training-objectives)
2. [Common Parameter Combinations](#common-parameter-combinations)
3. [Systematic Tuning Process](#systematic-tuning-process)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Performance vs Quality Trade-offs](#performance-vs-quality-trade-offs)
6. [Hardware-Specific Recommendations](#hardware-specific-recommendations)

## Training Objectives

### 1. Fast Prototyping and Development

**Goal**: Quick iteration and testing with minimal resource usage.

**Key Parameters**:
```yaml
dataset:
  target_recipes_per_cuisine: 1000
models:
  encoder:
    epochs: 5
    batch_size: 64
    learning_rate: 0.005
  palatenet:
    epochs: 10
    hidden_dim: 32
    num_layers: 1
```

**Expected Results**:
- Training time: 15-30 minutes
- Memory usage: 2-4GB
- Quality: Basic fusion capability
- Use case: Bug testing, initial development

### 2. Production-Quality Models

**Goal**: High-quality fusion recipes for deployment.

**Key Parameters**:
```yaml
dataset:
  target_recipes_per_cuisine: 10000
models:
  encoder:
    epochs: 50
    latent_dim: 512
    learning_rate: 0.0005
  palatenet:
    epochs: 100
    hidden_dim: 128
    num_layers: 3
    dropout: 0.3
```

**Expected Results**:
- Training time: 4-8 hours
- Memory usage: 8-16GB
- Quality: High-quality fusion with good diversity
- Spearman correlation: 0.35-0.45

### 3. Research and Experimentation

**Goal**: State-of-the-art results and novel insights.

**Key Parameters**:
```yaml
dataset:
  target_recipes_per_cuisine: 15000
  target_cuisines: ["japanese", "italian", "french", "chinese"]
models:
  encoder:
    epochs: 100
    latent_dim: 768
    learning_rate: 0.0003
  palatenet:
    epochs: 150
    hidden_dim: 256
    num_layers: 4
fusion:
  alpha_values: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

**Expected Results**:
- Training time: 12-24 hours
- Memory usage: 16-32GB
- Quality: Research-grade with extensive evaluation
- Spearman correlation: 0.45-0.60

### 4. Resource-Constrained Environments

**Goal**: Basic functionality on limited hardware.

**Key Parameters**:
```yaml
dataset:
  target_recipes_per_cuisine: 500
models:
  encoder:
    epochs: 3
    latent_dim: 64
    batch_size: 8
  palatenet:
    epochs: 5
    hidden_dim: 16
    num_layers: 1
training:
  device: "cpu"
  num_workers: 1
```

**Expected Results**:
- Training time: 10-20 minutes
- Memory usage: 1-2GB
- Quality: Basic fusion capability
- Hardware: Works on any modern laptop

## Common Parameter Combinations

### For Better Model Quality

**Increase Model Capacity**:
- `latent_dim`: 256 → 512 → 768
- `hidden_dim`: 64 → 128 → 256
- `num_layers`: 2 → 3 → 4

**Improve Training Stability**:
- `learning_rate`: 0.001 → 0.0005 → 0.0003
- `dropout`: 0.2 → 0.3 → 0.4
- `max_grad_norm`: 1.0 → 0.5 → 0.3

**Enhanced Data Quality**:
- `target_recipes_per_cuisine`: 5000 → 10000 → 15000
- `min_ingredient_frequency`: 10 → 15 → 20
- `image_size`: 256 → 384 → 512

### For Faster Training

**Reduce Model Size**:
- `latent_dim`: 256 → 128 → 64
- `hidden_dim`: 64 → 32 → 16
- `num_layers`: 2 → 1

**Accelerate Convergence**:
- `learning_rate`: 0.001 → 0.005 → 0.01
- `batch_size`: 32 → 64 → 128
- `epochs`: 20 → 10 → 5

**Reduce Data Processing**:
- `target_recipes_per_cuisine`: 5000 → 1000 → 500
- `image_size`: 256 → 128 → 64
- `text_max_length`: 512 → 256 → 128

### For Memory Efficiency

**Reduce Batch Sizes**:
- `encoder.batch_size`: 32 → 16 → 8
- `palatenet.batch_size`: 64 → 32 → 16

**Enable Memory Optimizations**:
- `mixed_precision`: true
- `gradient_checkpointing`: true
- `pin_memory`: false

**Reduce Model Dimensions**:
- `latent_dim`: 256 → 128
- `hidden_dim`: 64 → 32

## Systematic Tuning Process

### Phase 1: Baseline Establishment (Quick Prototype)

1. Start with `configs/quick_prototype.yaml`
2. Verify the pipeline works end-to-end
3. Note training time and basic quality metrics
4. Establish baseline Spearman correlation

### Phase 2: Scale Up (Production)

1. Use `configs/production.yaml` as starting point
2. Incrementally increase dataset size:
   - 1K → 2K → 5K → 10K recipes per cuisine
3. Monitor memory usage and training time
4. Track quality improvements

### Phase 3: Architecture Optimization

1. **Encoder Tuning**:
   - Test latent dimensions: [128, 256, 512]
   - Try learning rates: [0.001, 0.0005, 0.0003]
   - Adjust temperature: [0.05, 0.07, 0.1]

2. **PalateNet Tuning**:
   - Test hidden dimensions: [32, 64, 128, 256]
   - Try layer counts: [1, 2, 3, 4]
   - Adjust dropout: [0.1, 0.2, 0.3, 0.4]

### Phase 4: Advanced Optimization

1. **Learning Rate Scheduling**:
   - Test cosine annealing vs step decay
   - Tune warmup steps: [100, 500, 1000]

2. **Data Augmentation**:
   - Add color jitter for images
   - Try different text length limits

3. **Fusion Techniques**:
   - Experiment with alpha value ranges
   - Test different aggregation methods

## Troubleshooting Guide

### Training Loss Not Decreasing

**Symptoms**: Loss plateaus early or oscillates

**Solutions**:
1. Increase learning rate: 0.001 → 0.005
2. Reduce batch size: 64 → 32 → 16
3. Check data quality and preprocessing
4. Add learning rate warmup

**Parameter Changes**:
```yaml
models:
  encoder:
    learning_rate: 0.005
    batch_size: 16
training:
  warmup_steps: 500
```

### Overfitting (Training >> Validation Performance)

**Symptoms**: Large gap between train and validation metrics

**Solutions**:
1. Increase dropout: 0.2 → 0.4
2. Reduce model capacity
3. Add more training data
4. Implement early stopping

**Parameter Changes**:
```yaml
models:
  palatenet:
    dropout: 0.4
    hidden_dim: 32  # Reduce from 64
dataset:
  target_recipes_per_cuisine: 10000  # Increase
```

### Poor Fusion Quality

**Symptoms**: Fusion recipes lack creativity or cultural integration

**Solutions**:
1. Increase encoder training duration
2. Higher latent dimensions for better representations
3. More diverse training data
4. Adjust fusion alpha values

**Parameter Changes**:
```yaml
models:
  encoder:
    epochs: 50  # Increase from 20
    latent_dim: 512  # Increase from 256
fusion:
  alpha_values: [0.2, 0.4, 0.6, 0.8]  # More granular
```

### Memory Errors

**Symptoms**: CUDA out of memory or system crashes

**Solutions**:
1. Reduce batch sizes
2. Enable gradient checkpointing
3. Use mixed precision
4. Reduce model dimensions

**Parameter Changes**:
```yaml
models:
  encoder:
    batch_size: 16  # Reduce from 32
  palatenet:
    batch_size: 32  # Reduce from 64
training:
  mixed_precision: true
  gradient_checkpointing: true
```

### Low Spearman Correlation

**Symptoms**: Rating predictions are poor (<0.2 correlation)

**Solutions**:
1. Increase PalateNet capacity
2. More training epochs
3. Better ingredient frequency filtering
4. Add graph layers

**Parameter Changes**:
```yaml
models:
  palatenet:
    epochs: 100  # Increase from 50
    hidden_dim: 128  # Increase from 64
    num_layers: 3  # Increase from 2
dataset:
  min_ingredient_frequency: 15  # Increase from 10
```

## Performance vs Quality Trade-offs

### Speed vs Quality Matrix

| Configuration | Training Time | Memory Usage | Quality Score | Use Case |
|---------------|---------------|--------------|---------------|----------|
| Quick Prototype | 15-30 min | 2-4GB | ⭐⭐ | Development |
| Balanced | 1-2 hours | 4-8GB | ⭐⭐⭐ | Testing |
| Production | 4-8 hours | 8-16GB | ⭐⭐⭐⭐ | Deployment |
| Research | 12-24 hours | 16-32GB | ⭐⭐⭐⭐⭐ | Publication |

### Key Trade-off Parameters

**Most Impact on Quality**:
1. `target_recipes_per_cuisine` (data size)
2. `latent_dim` (representation capacity)
3. `epochs` (training duration)
4. `hidden_dim` (model capacity)

**Most Impact on Speed**:
1. `batch_size` (larger = faster)
2. `epochs` (fewer = faster)
3. `target_recipes_per_cuisine` (smaller = faster)
4. `image_size` (smaller = faster)

**Most Impact on Memory**:
1. `batch_size` (smaller = less memory)
2. `latent_dim` (smaller = less memory)
3. `image_size` (smaller = less memory)
4. `hidden_dim` (smaller = less memory)

## Hardware-Specific Recommendations

### High-End GPU (RTX 4090, A100)

```yaml
models:
  encoder:
    batch_size: 128
    latent_dim: 768
  palatenet:
    batch_size: 256
    hidden_dim: 256
training:
  mixed_precision: true
  num_workers: 12
```

### Mid-Range GPU (RTX 3070, RTX 4070)

```yaml
models:
  encoder:
    batch_size: 32
    latent_dim: 384
  palatenet:
    batch_size: 64
    hidden_dim: 128
training:
  mixed_precision: true
  num_workers: 8
```

### Budget GPU (GTX 1660, RTX 3060)

```yaml
models:
  encoder:
    batch_size: 16
    latent_dim: 256
  palatenet:
    batch_size: 32
    hidden_dim: 64
training:
  mixed_precision: true
  gradient_checkpointing: true
  num_workers: 4
```

### CPU Only

```yaml
models:
  encoder:
    batch_size: 8
    latent_dim: 128
  palatenet:
    batch_size: 16
    hidden_dim: 32
training:
  device: "cpu"
  num_workers: 2
  mixed_precision: false
```

## Best Practices Summary

1. **Start Small**: Always begin with quick_prototype.yaml
2. **Incremental Scaling**: Increase one parameter group at a time
3. **Monitor Resources**: Track memory and time usage
4. **Validate Quality**: Check fusion results manually
5. **Document Changes**: Keep track of what works
6. **Use Templates**: Leverage provided configuration templates
7. **Regular Evaluation**: Run evaluation metrics frequently
8. **Hardware Awareness**: Match configuration to available resources