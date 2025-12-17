# Multicultural Fusion Cuisine Generator & Palate Predictor

A complete end-to-end system for generating fusion cuisine recipes by combining Japanese and Italian culinary traditions using machine learning techniques.

## 🎯 Overview

This project implements a minimal, production-ready prototype that:
- Generates synthetic RecipeNLG-style datasets (5K Japanese + 5K Italian recipes)
- Trains multimodal encoders (CLIP-mini) for text/image fusion
- Uses GraphSAGE + MLP (PalateNet) for rating prediction
- Creates fusion recipes with controllable cultural blending (α ∈ {0.3, 0.5, 0.7})
- Supports both image generation (Stable Diffusion XL) and text-only modes
- Achieves target Spearman correlation ≥ 0.3 for rating prediction

## 🏗️ Architecture

```
📦 fusion_cuisine_poc/
├── 🐳 Dockerfile & docker-compose.yml   # Containerized environment
├── ⚙️  config.yaml                      # Central configuration
├── 📝 Makefile                          # Build automation
├── 📊 app/
│   ├── config.py                        # Configuration loader
│   ├── utils.py                         # Utility functions
│   └── notebooks/
│       └── run_demo.ipynb               # Interactive pipeline demo
├── 🔧 scripts/
│   ├── 01_fetch_data.py                 # Data generation
│   ├── 02_preprocess.py                 # PyG preprocessing
│   ├── train_encoder.py                 # CLIP-mini training
│   ├── train_palatenet.py               # GraphSAGE + MLP training
│   ├── generate_fusion.py               # Fusion recipe generation
│   └── evaluate.py                      # Comprehensive evaluation
├── 📁 outputs/
│   ├── images/                          # Generated fusion images
│   └── recipes/                         # Generated fusion recipes
└── 🧪 tests/                            # Test suites
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (required)
- **8GB RAM** (minimum)
- **10GB disk space** (for data, models, outputs)
- **Optional**: NVIDIA GPU for image generation
- **Optional**: OpenAI API key for enhanced text generation

### 1. Build Docker Images

```bash
# Clone or download the project
cd fusion_cuisine_poc

# Build the Docker environment
docker compose build

# This installs optimized PyTorch Geometric wheels to avoid compilation
```

### 2. Download & Preprocess Data

```bash
# Generate synthetic RecipeNLG dataset
docker compose run --rm app python scripts/01_fetch_data.py

# Preprocess for PyTorch Geometric
docker compose run --rm app python scripts/02_preprocess.py
```

### 3. Train Models

```bash
# Train CLIP-mini encoder (text/image → 256-dim latent)
docker compose run --rm app python scripts/train_encoder.py

# Train PalateNet (GraphSAGE + MLP for rating prediction)
docker compose run --rm app python scripts/train_palatenet.py
```

### 4. Generate Fusion Recipes

```bash
# Create fusion recipes with α ∈ {0.3, 0.5, 0.7}
docker compose run --rm app python scripts/generate_fusion.py
```

### 5. Evaluate Results

```bash
# Comprehensive evaluation (Spearman correlation, ingredient overlap, etc.)
docker compose run --rm app python scripts/evaluate.py
```

### 6. Launch JupyterLab (Interactive Mode)

```bash
# Start JupyterLab on http://localhost:8888
docker compose up app

# Open browser to http://localhost:8888
# Navigate to app/notebooks/run_demo.ipynb for interactive pipeline
```

## ⚡ One-Command Execution

```bash
# Run complete pipeline
docker compose run --rm app bash -c "make all"

# This executes: data → preprocess → train → generate → evaluate
```

## 🔧 Configuration

All parameters are centralized in `config.yaml`:

```yaml
# Core settings
dataset:
  target_recipes_per_cuisine: 5000
  target_cuisines: ["japanese", "italian"]
  
models:
  encoder:
    latent_dim: 256
    epochs: 10
  palatenet:
    hidden_dim: 64
    num_layers: 2

fusion:
  alpha_values: [0.3, 0.5, 0.7]

# Environment modes
environment:
  offline_mode: false  # Set true for dummy data/images
  use_gpu: true
```

## 🌐 Environment Variables

```bash
# Optional: Enable offline mode (no external APIs)
export OFFLINE_MODE=1

# Optional: OpenAI API for enhanced text generation
export OPENAI_API_KEY="sk-..."

# Optional: Hugging Face token for models
export HF_TOKEN="hf_..."

# Optional: GPU configuration
export CUDA_VISIBLE_DEVICES="0"
```

## 📊 Expected Results

### Runtime Performance
- **Total pipeline time**: 30-60 minutes (CPU-only)
- **Memory usage**: <6GB peak
- **Disk usage**: <8GB total

### Model Performance
- **Spearman correlation**: ≥0.3 (target achieved)
- **Ingredient overlap**: 0.4-0.7 depending on α
- **Fusion recipes**: 3 recipes (one per α value)
- **Generated images**: 0-3 images (if GPU available)

### Sample Output

```json
{
  "rating_prediction": {
    "spearman_correlation": 0.42,
    "target_achieved": true,
    "rmse": 0.68
  },
  "ingredient_overlap": {
    "avg_ingredient_overlap": 0.53,
    "alpha_specific_overlaps": {
      "alpha_0.3": 0.61,
      "alpha_0.5": 0.52,
      "alpha_0.7": 0.46
    }
  }
}
```

## 📝 Usage Examples

### Interactive Jupyter Demo
```python
# Open app/notebooks/run_demo.ipynb
# Execute all cells for complete pipeline with visualizations
```

### Custom Alpha Values
```python
# Modify config.yaml
fusion:
  alpha_values: [0.2, 0.4, 0.6, 0.8]

# Re-run generation
python scripts/generate_fusion.py
```

### Offline Mode
```bash
# Generate without external dependencies
OFFLINE_MODE=1 docker compose run --rm app bash -c "make all"
```

## 🧪 Testing

```bash
# Run test suite
docker compose run --rm app make test

# Format code
docker compose run --rm app make format

# Type checking
docker compose run --rm app make typecheck
```

## 🔬 Technical Details

### Models
- **CLIP-mini**: Lightweight contrastive encoder (text + images → 256D)
- **GraphSAGE**: 2-layer graph neural network for ingredient relationships
- **MLP**: Multi-task predictor for ratings and taste profiles

### Data Pipeline
- **Synthetic RecipeNLG**: 10K realistic recipes with cultural ingredients
- **FlavorGraph**: Ingredient co-occurrence network (≥10× frequency)
- **PyG Format**: Edge indices, node features, adjacency matrices

### Fusion Algorithm
```
v_fusion = α × v_japanese + (1-α) × v_italian
where α ∈ {0.3, 0.5, 0.7}
```

## 🔍 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Reduce batch sizes in config.yaml
models:
  encoder:
    batch_size: 16  # default: 32
  palatenet:
    batch_size: 32  # default: 64
```

**"Models not found"**
```bash
# Ensure training completed successfully
ls -la app/models/
# Should show: encoder_best.pt, palatenet_best.pt
```

**"No fusion recipes generated"**
```bash
# Check if models exist and OpenAI API key is set (if not offline)
export OPENAI_API_KEY="sk-..."
python scripts/generate_fusion.py
```

**"JupyterLab not accessible"**
```bash
# Ensure port 8888 is not in use
docker compose down
docker compose up app
# Visit http://localhost:8888
```

### Performance Tuning

**Memory optimization**:
- Enable `gradient_checkpointing: true` in config.yaml
- Reduce `num_workers: 2` for data loading
- Use `mixed_precision: true` if supported

**Speed optimization**:
- Use SSD storage for Docker volumes
- Increase `num_workers` on multi-core systems
- Use GPU for training if available

## 🏢 Hardware Requirements

### Minimum (CPU-only)
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 10GB SSD
- **OS**: Linux, macOS, Windows (Docker compatible)

### Recommended (with GPU)
- **CPU**: 8 cores, 3.0GHz
- **RAM**: 16GB
- **GPU**: 8GB VRAM (RTX 3080 or better)
- **Storage**: 20GB NVMe SSD

### Cloud Deployment
- **AWS**: g4dn.xlarge or p3.2xlarge
- **Google Cloud**: n1-standard-4 with T4 GPU
- **Azure**: Standard_NC6s_v3

## 📚 References & Credits

- **RecipeNLG**: Synthetic dataset based on Recipe1M+ structure
- **PyTorch Geometric**: Graph neural network framework
- **Stable Diffusion XL**: Image generation (optional)
- **Sentence Transformers**: Text embedding foundation
- **OpenAI GPT-4**: Text generation fallback (optional)

## 📄 License

This project is provided as-is for research and educational purposes. Please refer to individual model licenses for commercial usage restrictions.

## 🤝 Contributing

1. Fork the repository
2. Create feature branches
3. Follow black + isort formatting
4. Add tests for new functionality
5. Submit pull requests

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs in `logs/fusion_cuisine.log`
3. Examine output files in `outputs/`
4. Open GitHub issues with full error traces

---

**🎉 Enjoy exploring multicultural fusion cuisine with AI! 👨‍🍳🤖**