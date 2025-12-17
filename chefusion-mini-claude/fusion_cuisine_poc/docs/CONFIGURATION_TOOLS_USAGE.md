# Configuration Tools Usage Guide

This guide explains how to use the configuration management tools for the Fusion Cuisine Learning System.

## Available Tools

1. **validate_config.py** - Validates configuration files for errors and compatibility
2. **estimate_resources.py** - Estimates memory, time, and disk requirements
3. **compare_configs.py** - Compares two configurations and analyzes differences

## Tool 1: Configuration Validation

### Purpose
Validates configuration files for parameter validity, resource requirements, and compatibility issues before training.

### Basic Usage
```bash
# Validate the main config
python scripts/validate_config.py config.yaml

# Validate a template config
python scripts/validate_config.py configs/production.yaml

# Quiet mode (only show errors)
python scripts/validate_config.py config.yaml --quiet
```

### Example Output
```
==========================================
CONFIGURATION VALIDATION RESULTS
==========================================

✅ Configuration is valid with no issues!

📊 ESTIMATES:
  • Estimated memory usage: 6.2 GB
  • Estimated training time: 45 minutes
==========================================
```

### What It Checks
- **Required parameters**: Ensures all mandatory configuration sections exist
- **Value ranges**: Validates parameters are within acceptable ranges
- **Type checking**: Confirms parameters have correct data types
- **Compatibility**: Checks for incompatible parameter combinations
- **Resource estimation**: Estimates if configuration fits within limits

### Common Validation Errors
```bash
❌ ERRORS (3):
  • Required dataset parameter 'target_recipes_per_cuisine' missing
  • encoder.batch_size must be positive integer
  • palatenet.dropout must be between 0 and 1

⚠️  WARNINGS (2):
  • target_recipes_per_cuisine < 100 may give poor results
  • encoder.learning_rate > 0.1 may cause training instability
```

## Tool 2: Resource Estimation

### Purpose
Estimates computational requirements (memory, time, disk space) based on configuration parameters.

### Basic Usage
```bash
# Basic estimation
python scripts/estimate_resources.py config.yaml

# Detailed breakdown
python scripts/estimate_resources.py configs/production.yaml --detailed

# JSON output for automation
python scripts/estimate_resources.py config.yaml --json
```

### Example Output
```
======================================================================
RESOURCE ESTIMATION REPORT
======================================================================

📊 MEMORY REQUIREMENTS:
  Total Estimated Memory: 6.2 GB

⏱️  TRAINING TIME:
  Total Estimated Time: 2.3 hours

💾 DISK SPACE:
  Total Estimated Space: 4.8 GB

🖥️  HARDWARE CATEGORY: Mid-range Desktop
  Recommendations:
    • CPU: 6+ cores, 3.0+ GHz
    • MEMORY: 16GB+ RAM
    • GPU: RTX 3060 or better
    • STORAGE: 20GB+ SSD space
======================================================================
```

### Detailed Breakdown
```bash
python scripts/estimate_resources.py configs/production.yaml --detailed
```

Shows breakdown by component:
- Data Memory: Image storage, text embeddings, graph data
- Model Memory: Neural network parameters
- Training Overhead: Gradients, optimizer states
- Time Breakdown: Preprocessing, training phases, evaluation

### JSON Output for Automation
```bash
python scripts/estimate_resources.py config.yaml --json
```

Outputs structured data for scripts:
```json
{
  "memory": {
    "total_memory": 6.2,
    "data_memory": 2.1,
    "model_memory": 1.8,
    ...
  },
  "training_time": {
    "total_time": 138.5,
    "encoder_training": 45.2,
    ...
  }
}
```

## Tool 3: Configuration Comparison

### Purpose
Compares two configuration files, highlights differences, and estimates performance impact.

### Basic Usage
```bash
# Compare two configs
python scripts/compare_configs.py config1.yaml config2.yaml

# Compare with impact analysis
python scripts/compare_configs.py configs/quick_prototype.yaml configs/production.yaml --impact

# Show only high-impact changes
python scripts/compare_configs.py old_config.yaml new_config.yaml --high-only

# JSON output
python scripts/compare_configs.py config1.yaml config2.yaml --json
```

### Example Output
```
================================================================================
CONFIGURATION COMPARISON: quick_prototype.yaml vs production.yaml
================================================================================

📊 SUMMARY:
  Total differences: 12
  High impact:       5
  Medium impact:     4
  Low impact:        3

🔴 HIGH IMPACT DIFFERENCES (5):

  Parameter: dataset.target_recipes_per_cuisine
  quick_prototype.yaml: 1000
  production.yaml: 10000
  Impact: Dataset size per cuisine - affects model quality and training time

  Parameter: models.encoder.latent_dim
  quick_prototype.yaml: 128
  production.yaml: 512
  Impact: Embedding dimension - affects representation quality and memory usage

🎯 PERFORMANCE IMPACT ANALYSIS:
  Estimated changes:
    • Training Time: Significantly longer (~10.0x)
    • Memory Usage: Higher (~4.0x more memory)
    • Quality: Likely better due to more data

💡 RECOMMENDATIONS:
  ⚠️  High-impact changes detected - test thoroughly before production
  💾 Consider enabling mixed_precision and gradient_checkpointing for memory efficiency
================================================================================
```

### Impact Analysis Features
- **Impact categorization**: High/Medium/Low impact classification
- **Performance estimation**: Predicted changes in memory, time, quality
- **Automatic recommendations**: Suggestions based on detected changes
- **Risk assessment**: Warnings about potentially problematic configurations

## Practical Workflows

### 1. Before Training: Configuration Validation
```bash
# 1. Validate your configuration
python scripts/validate_config.py config.yaml

# 2. Check if resources are sufficient
python scripts/estimate_resources.py config.yaml --detailed

# 3. If using a template, compare with your needs
python scripts/compare_configs.py configs/production.yaml config.yaml --impact
```

### 2. Optimization: Finding the Right Configuration
```bash
# Start with quick prototype
cp configs/quick_prototype.yaml my_config.yaml

# Test that it works
python scripts/validate_config.py my_config.yaml

# Gradually scale up and check impact
python scripts/compare_configs.py configs/quick_prototype.yaml configs/production.yaml --impact

# Customize parameters and validate
# ... edit my_config.yaml ...
python scripts/validate_config.py my_config.yaml
```

### 3. Experiment Tracking: Compare Different Attempts
```bash
# Save configs with descriptive names
cp config.yaml experiments/config_baseline_v1.yaml
# ... train and evaluate ...

# Try new parameters
# ... edit config.yaml ...
cp config.yaml experiments/config_larger_model_v2.yaml

# Compare experiments
python scripts/compare_configs.py experiments/config_baseline_v1.yaml experiments/config_larger_model_v2.yaml --impact
```

### 4. Production Deployment: Final Validation
```bash
# Validate production config thoroughly
python scripts/validate_config.py production_config.yaml

# Ensure resources are available
python scripts/estimate_resources.py production_config.yaml --detailed

# Compare against development config to understand changes
python scripts/compare_configs.py dev_config.yaml production_config.yaml --impact
```

## Integration with Main Pipeline

### Using Makefile
Add validation to your Makefile:
```makefile
validate-config:
	python scripts/validate_config.py config.yaml

estimate-resources:
	python scripts/estimate_resources.py config.yaml --detailed

.PHONY: validate-config estimate-resources
```

### Pre-training Checks
```bash
# Run all checks before training
make validate-config
make estimate-resources
make all  # Run the main pipeline
```

### Automated Workflows
```bash
#!/bin/bash
# training_pipeline.sh

# Validate configuration
python scripts/validate_config.py config.yaml
if [ $? -ne 0 ]; then
    echo "Configuration validation failed!"
    exit 1
fi

# Check resources
python scripts/estimate_resources.py config.yaml --json > resource_estimates.json

# Continue with training...
make all
```

## Configuration Templates Usage

### Quick Start Templates
```bash
# For fast prototyping
cp configs/quick_prototype.yaml config.yaml

# For production quality
cp configs/production.yaml config.yaml

# For research experiments
cp configs/research.yaml config.yaml

# For limited hardware
cp configs/resource_constrained.yaml config.yaml
```

### Template Customization
1. **Start with appropriate template**: Choose based on your goals and hardware
2. **Validate the template**: `python scripts/validate_config.py configs/template.yaml`
3. **Customize parameters**: Edit the copy for your specific needs
4. **Validate changes**: `python scripts/validate_config.py config.yaml`
5. **Compare with original**: `python scripts/compare_configs.py configs/template.yaml config.yaml --impact`

## Troubleshooting

### Common Issues

**Configuration not found**:
```bash
python scripts/validate_config.py nonexistent.yaml
# Error: Config file not found: nonexistent.yaml
```

**Invalid YAML syntax**:
```bash
# Error: Failed to validate config: yaml.scanner.ScannerError
```
Fix: Check YAML syntax, proper indentation, matching quotes

**Resource estimation seems wrong**:
- Estimates are rough approximations
- Actual performance depends on hardware
- Use estimates for relative comparison, not absolute values

**Validation passes but training fails**:
- Validation checks basic parameter validity
- Some issues only surface during actual training
- Use quick prototype configs to test full pipeline first

## Best Practices

1. **Always validate before training**: Catch issues early
2. **Use resource estimation**: Plan hardware and time requirements
3. **Compare configurations**: Understand impact of changes
4. **Start with templates**: Build on proven configurations
5. **Document experiments**: Keep track of configuration changes
6. **Validate incrementally**: Check each modification as you make it