# Configuration Guide: Fusion Cuisine Learning System

This guide provides comprehensive documentation for all configurable parameters in the multicultural fusion cuisine generator and their effects on learning behavior and results.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Architecture Parameters](#model-architecture-parameters)
3. [Training Hyperparameters](#training-hyperparameters)
4. [Data Configuration](#data-configuration)
5. [Fusion Parameters](#fusion-parameters)
6. [Advanced Parameters](#advanced-parameters)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

## Quick Start

The main configuration file is `config.yaml`. Most parameters can be modified without touching the code. Here are the most impactful parameters for quick experimentation:

```yaml
# Quick tuning parameters
models:
  encoder:
    epochs: 20          # Increase for better quality (default: 10)
    learning_rate: 0.005 # Increase for faster learning (default: 0.001)
  palatenet:
    epochs: 50          # Increase for better ratings (default: 20)
    hidden_dim: 128     # Increase for more model capacity (default: 64)

dataset:
  target_recipes_per_cuisine: 10000  # More data = better results (default: 5000)

fusion:
  alpha_values: [0.2, 0.4, 0.6, 0.8]  # More granular fusion control
```

## Model Architecture Parameters

### Encoder Model (CLIP-mini)

| Parameter | Location | Default | Effect | Recommended Range |
|-----------|----------|---------|--------|-------------------|
| `latent_dim` | `models.encoder.latent_dim` | 256 | Size of shared embedding space. Higher = more expressive but slower | 128-512 |
| `projection_dim` | `models.encoder.projection_dim` | 128 | Projection head dimension for contrastive learning | 64-256 |
| `temperature` | `models.encoder.temperature` | 0.1 | Contrastive loss temperature. Lower = harder negative mining | 0.05-0.2 |

**Effects on Learning:**
- **Higher `latent_dim`**: Better representation quality but requires more training time and memory
- **Lower `temperature`**: More discriminative embeddings but harder optimization
- **Higher `projection_dim`**: Better contrastive learning but slower training

### PalateNet Model (GraphSAGE + MLP)

| Parameter | Location | Default | Effect | Recommended Range |
|-----------|----------|---------|--------|-------------------|
| `hidden_dim` | `models.palatenet.hidden_dim` | 64 | Hidden layer size. Higher = more model capacity | 32-256 |
| `num_layers` | `models.palatenet.num_layers` | 2 | Graph network depth. Higher = more complex ingredient relationships | 1-4 |
| `dropout` | `models.palatenet.dropout` | 0.2 | Regularization strength. Higher = less overfitting | 0.0-0.5 |
| `aggregation` | `models.palatenet.aggregation` | "mean" | Graph aggregation method | "mean", "max", "sum" |

**Effects on Learning:**
- **Higher `hidden_dim`**: Better modeling of complex ingredient interactions
- **More `num_layers`**: Captures longer-range ingredient dependencies but may overfit
- **Higher `dropout`**: Prevents overfitting but may hurt performance with small datasets

## Training Hyperparameters

### Learning Rates

| Parameter | Location | Default | Effect | When to Adjust |
|-----------|----------|---------|--------|----------------|
| `encoder.learning_rate` | `models.encoder.learning_rate` | 0.001 | Encoder training speed | Increase if loss plateaus early, decrease if loss oscillates |
| `palatenet.learning_rate` | `models.palatenet.learning_rate` | 0.001 | PalateNet training speed | Increase for faster convergence, decrease for stability |

**Learning Rate Guidelines:**
- **Fast experimentation**: 0.005-0.01
- **Stable training**: 0.001-0.003
- **Fine-tuning**: 0.0001-0.0005

### Batch Sizes

| Parameter | Location | Default | Effect | Hardware Considerations |
|-----------|----------|---------|--------|------------------------|
| `encoder.batch_size` | `models.encoder.batch_size` | 32 | Memory usage and gradient noise | GPU memory limited |
| `palatenet.batch_size` | `models.palatenet.batch_size` | 64 | Training stability | Can be larger for graph data |

**Batch Size Guidelines:**
- **8GB RAM**: Keep encoder batch_size ≤ 32, palatenet ≤ 64
- **16GB RAM**: Can increase to 64 and 128 respectively
- **Small datasets**: Use smaller batches (16-32) for better generalization

### Training Epochs

| Parameter | Location | Default | Effect | Signs of Optimal Value |
|-----------|----------|---------|--------|----------------------|
| `encoder.epochs` | `models.encoder.epochs` | 10 | Encoder training duration | Stop when validation loss plateaus |
| `palatenet.epochs` | `models.palatenet.epochs` | 20 | PalateNet training duration | Monitor Spearman correlation |

**Epoch Guidelines:**
- **Quick prototyping**: 5-10 epochs
- **Production quality**: 20-50 epochs
- **Research/experimentation**: 50-100 epochs

## Data Configuration

### Dataset Parameters

| Parameter | Location | Default | Effect | Recommendations |
|-----------|----------|---------|--------|-----------------|
| `target_recipes_per_cuisine` | `dataset.target_recipes_per_cuisine` | 5000 | Dataset size per cuisine | More data = better results, but slower training |
| `target_cuisines` | `dataset.target_cuisines` | ["japanese", "italian"] | Cuisine types to include | Can add "french", "chinese", "mexican", etc. |
| `min_ingredient_frequency` | `dataset.min_ingredient_frequency` | 10 | Minimum times an ingredient appears | Higher = smaller vocabulary, lower = more diversity |

**Data Size Impact:**
- **1K recipes/cuisine**: Fast prototyping, basic fusion
- **5K recipes/cuisine**: Good balance for development
- **10K+ recipes/cuisine**: Production quality, requires more training time

### Data Splits

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `test_split_ratio` | `dataset.test_split_ratio` | 0.1 | Fraction for final evaluation |
| `validation_split_ratio` | `dataset.validation_split_ratio` | 0.1 | Fraction for hyperparameter tuning |

## Fusion Parameters

### Alpha Blending

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `alpha_values` | `fusion.alpha_values` | [0.3, 0.5, 0.7] | Cultural blending ratios |
| `num_samples_per_alpha` | `fusion.num_samples_per_alpha` | 10 | Recipes generated per alpha |

**Alpha Value Guidelines:**
- **0.1-0.3**: Subtle fusion, mostly base cuisine
- **0.4-0.6**: Balanced fusion
- **0.7-0.9**: Strong fusion, mostly target cuisine

### Generation Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `generation.temperature` | `models.generation.temperature` | 0.7 | Creativity vs consistency in text generation |
| `generation.max_tokens` | `models.generation.max_tokens` | 1000 | Maximum recipe length |

## Advanced Parameters

### Training Optimization

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `mixed_precision` | `training.mixed_precision` | false | Memory efficiency and speed |
| `gradient_checkpointing` | `training.gradient_checkpointing` | true | Memory efficiency |
| `max_grad_norm` | `training.max_grad_norm` | 1.0 | Gradient clipping for stability |

### Environment Settings

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `device` | `training.device` | "auto" | Hardware acceleration |
| `num_workers` | `environment.num_workers` | 4 | Data loading parallelism |
| `random_seed` | `environment.random_seed` | 42 | Reproducibility |

## Performance Tuning

### For Speed Optimization

```yaml
# Fast training configuration
models:
  encoder:
    epochs: 5
    batch_size: 64
  palatenet:
    epochs: 10
    hidden_dim: 32
    num_layers: 1

dataset:
  target_recipes_per_cuisine: 1000

training:
  mixed_precision: true
  num_workers: 8
```

### For Quality Optimization

```yaml
# High-quality training configuration
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

dataset:
  target_recipes_per_cuisine: 10000

fusion:
  alpha_values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  num_samples_per_alpha: 20
```

### For Memory-Constrained Environments

```yaml
# Low-memory configuration
models:
  encoder:
    batch_size: 16
    latent_dim: 128
  palatenet:
    batch_size: 32
    hidden_dim: 32

training:
  gradient_checkpointing: true
  mixed_precision: true

environment:
  num_workers: 2
```

## Troubleshooting

### Common Issues and Solutions

**Training Loss Not Decreasing:**
- Increase learning rate (0.001 → 0.005)
- Decrease batch size
- Check data quality

**Overfitting (Train >> Validation Performance):**
- Increase dropout (0.2 → 0.4)
- Decrease model capacity (hidden_dim)
- Add more training data

**Memory Errors:**
- Decrease batch sizes
- Enable gradient checkpointing
- Use mixed precision
- Reduce model dimensions

**Poor Fusion Quality:**
- Increase encoder epochs
- Higher latent dimensions
- More diverse training data
- Adjust alpha values

**Low Spearman Correlation:**
- Increase PalateNet epochs
- Higher hidden dimensions
- More graph layers
- Better ingredient frequency filtering

### Monitoring Training

Key metrics to watch:
- **Encoder**: Contrastive loss should decrease steadily
- **PalateNet**: Validation Spearman correlation should increase
- **Memory**: Stay below resource limits
- **Time**: Balance quality vs training duration

### Configuration Validation

Before training, validate your configuration:

```bash
# Check configuration validity
python scripts/validate_config.py config.yaml

# Estimate resource requirements
python scripts/estimate_resources.py config.yaml
```

## Example Configurations

See the `configs/` directory for pre-built configurations:
- `configs/quick_prototype.yaml` - Fast development
- `configs/production.yaml` - High-quality results
- `configs/research.yaml` - Extensive experimentation
- `configs/resource_constrained.yaml` - Limited hardware

## Best Practices

1. **Start small**: Begin with quick_prototype.yaml
2. **Monitor metrics**: Watch both loss and evaluation metrics
3. **Iterative tuning**: Change one parameter group at a time
4. **Document experiments**: Keep track of configuration changes
5. **Validate results**: Test fusion quality manually
6. **Resource planning**: Estimate time and memory before training