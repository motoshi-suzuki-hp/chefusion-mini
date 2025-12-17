.PHONY: all setup clean data preprocess train generate evaluate test format help jupyter validate-config fix-images monitor-logs results quick-results explore-recipes

# Default target - runs the complete pipeline
all: setup data preprocess train generate evaluate
	@echo "✅ Complete pipeline finished successfully!"
	@echo "📊 Results saved to outputs/ directory"
	@echo "📓 Check app/notebooks/ for interactive analysis"

# Setup environment and dependencies
setup:
	@echo "🔧 Setting up environment..."
	@mkdir -p app/{data,models,notebooks}
	@mkdir -p outputs/{images,recipes}
	@python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true
	@echo "✅ Environment setup complete"

# Download and prepare data
data:
	@echo "📥 Downloading and preparing data..."
	@python scripts/01_fetch_data.py
	@echo "✅ Data preparation complete"

# Preprocess data for training
preprocess:
	@echo "🔄 Preprocessing data..."
	@python scripts/02_preprocess.py
	@echo "✅ Preprocessing complete"

# Train all models
train: train-encoder train-palatenet
	@echo "✅ All model training complete"

# Train encoder model
train-encoder:
	@echo "🧠 Training encoder model..."
	@python scripts/train_encoder.py
	@echo "✅ Encoder training complete"

# Train PalateNet model
train-palatenet:
	@echo "🧠 Training PalateNet model..."
	@python scripts/train_palatenet.py
	@echo "✅ PalateNet training complete"

# Generate fusion recipes
generate:
	@echo "🎨 Generating fusion recipes..."
	@python scripts/generate_fusion.py
	@echo "✅ Fusion generation complete"

# Evaluate results
evaluate:
	@echo "📊 Evaluating results..."
	@python scripts/evaluate.py
	@echo "✅ Evaluation complete"

# Run tests
test:
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v --tb=short
	@echo "✅ Tests complete"

# Format code
format:
	@echo "🎨 Formatting code..."
	@python -m black app/ scripts/ tests/ --line-length 88
	@python -m isort app/ scripts/ tests/ --profile black
	@echo "✅ Code formatting complete"

# Type check
typecheck:
	@echo "🔍 Type checking..."
	@python -m mypy app/ scripts/ --ignore-missing-imports
	@echo "✅ Type checking complete"

# Start JupyterLab
jupyter:
	@echo "🚀 Starting JupyterLab..."
	@echo "📓 JupyterLab will be available at http://localhost:8888"
	@jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

# Quick demo - minimal pipeline for testing
demo:
	@echo "🎬 Running quick demo..."
	@OFFLINE_MODE=1 python scripts/01_fetch_data.py
	@OFFLINE_MODE=1 python scripts/02_preprocess.py
	@echo "✅ Demo complete - check outputs/ for results"

# Validate configuration
validate-config:
	@echo "🔍 Validating configuration..."
	@python scripts/validate_config.py config.yaml
	@echo "✅ Configuration validation complete"

# Fix image processing warnings
fix-images:
	@echo "🖼️  Fixing image processing issues..."
	@python scripts/fix_image_warnings.py
	@echo "✅ Image fixes complete"

# Monitor server logs in real-time
monitor-logs:
	@echo "🔍 Starting log monitoring..."
	@python scripts/monitor_logs.py

# Monitor logs with custom settings
monitor-logs-custom:
	@echo "🔍 Starting custom log monitoring..."
	@python scripts/monitor_logs.py --error-threshold 5 --warning-threshold 10 --stats-interval 15

# Show comprehensive results
results:
	@echo "📊 Displaying project results..."
	@python scripts/show_results.py

# Quick results overview
quick-results:
	@echo "⚡ Quick results check..."
	@python scripts/quick_results.py

# Explore recipes interactively
explore-recipes:
	@echo "🍽️ Starting recipe explorer..."
	@python scripts/recipe_explorer.py --interactive

# Export results to HTML
export-results:
	@echo "📄 Exporting results to HTML..."
	@python scripts/show_results.py --format html --export

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf app/data/*
	@rm -rf app/models/*
	@rm -rf outputs/*
	@rm -rf __pycache__/ .pytest_cache/ .mypy_cache/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Show help
help:
	@echo "Fusion Cuisine POC - Available Commands:"
	@echo ""
	@echo "  🎯 Primary Commands:"
	@echo "    make all          - Run complete pipeline (data → train → generate → evaluate)"
	@echo "    make jupyter      - Start JupyterLab server on port 8888"
	@echo "    make demo         - Quick demo with offline mode"
	@echo ""
	@echo "  🔧 Individual Steps:"
	@echo "    make setup        - Setup environment and dependencies"
	@echo "    make data         - Download and prepare data"
	@echo "    make preprocess   - Preprocess data for training"
	@echo "    make train        - Train all models"
	@echo "    make generate     - Generate fusion recipes"
	@echo "    make evaluate     - Evaluate results"
	@echo ""
	@echo "  🧪 Development:"
	@echo "    make test         - Run test suite"
	@echo "    make format       - Format code with black & isort"
	@echo "    make typecheck    - Run type checking"
	@echo "    make validate-config - Validate configuration file"
	@echo "    make fix-images   - Fix image processing warnings"
	@echo "    make monitor-logs - Monitor server logs in real-time"
	@echo "    make monitor-logs-custom - Monitor logs with custom settings"
	@echo "    make clean        - Clean up generated files"
	@echo ""
	@echo "  📊 Results & Analysis:"
	@echo "    make results      - Show comprehensive project results"
	@echo "    make quick-results - Quick overview of project status"
	@echo "    make explore-recipes - Interactive recipe browser"
	@echo "    make export-results - Export results to HTML report"
	@echo ""
	@echo "  🐳 Docker Usage:"
	@echo "    docker compose up app      - Start with JupyterLab"
	@echo "    docker compose run pipeline bash -c 'make all'"
	@echo ""
	@echo "  🌐 Environment Variables:"
	@echo "    OFFLINE_MODE=1    - Use offline/dummy data"
	@echo "    OPENAI_API_KEY    - OpenAI API key for text generation"
	@echo "    HF_TOKEN          - Hugging Face token for model access"