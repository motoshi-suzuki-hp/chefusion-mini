#!/usr/bin/env python3
"""
Configuration Validation Script for Fusion Cuisine POC

This script validates configuration files for parameter validity,
resource requirements, and compatibility.

Usage:
    python scripts/validate_config.py config.yaml
    python scripts/validate_config.py configs/production.yaml
"""

import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings


class ConfigValidator:
    """Validates fusion cuisine configuration files."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
    def validate(self, config_path: str) -> bool:
        """Validate a configuration file."""
        try:
            config = self._load_config(config_path)
            
            # Core validation checks
            self._validate_structure(config)
            self._validate_dataset_config(config.get('dataset', {}))
            self._validate_model_config(config.get('models', {}))
            self._validate_training_config(config.get('training', {}))
            self._validate_fusion_config(config.get('fusion', {}))
            self._validate_resource_limits(config.get('resources', {}))
            
            # Compatibility checks
            self._check_parameter_compatibility(config)
            self._estimate_resources(config)
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.errors.append(f"Failed to validate config: {e}")
            return False
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and parse configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if path.suffix not in ['.yaml', '.yml']:
            raise ValueError("Config file must be YAML format (.yaml or .yml)")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a dictionary at root level")
        
        return config
    
    def _validate_structure(self, config: Dict[str, Any]):
        """Validate the overall structure of the configuration."""
        required_sections = ['dataset', 'models', 'training', 'fusion']
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Required section '{section}' missing from config")
    
    def _validate_dataset_config(self, dataset: Dict[str, Any]):
        """Validate dataset configuration parameters."""
        if not dataset:
            return
        
        # Required parameters
        required = ['target_recipes_per_cuisine', 'target_cuisines']
        for param in required:
            if param not in dataset:
                self.errors.append(f"Required dataset parameter '{param}' missing")
        
        # Value validation
        if 'target_recipes_per_cuisine' in dataset:
            recipes = dataset['target_recipes_per_cuisine']
            if not isinstance(recipes, int) or recipes <= 0:
                self.errors.append("target_recipes_per_cuisine must be positive integer")
            elif recipes < 100:
                self.warnings.append("target_recipes_per_cuisine < 100 may give poor results")
            elif recipes > 50000:
                self.warnings.append("target_recipes_per_cuisine > 50000 requires significant resources")
        
        if 'target_cuisines' in dataset:
            cuisines = dataset['target_cuisines']
            if not isinstance(cuisines, list) or len(cuisines) == 0:
                self.errors.append("target_cuisines must be non-empty list")
            elif len(cuisines) > 6:
                self.warnings.append("More than 6 cuisines may require significant training time")
        
        if 'image_size' in dataset:
            size = dataset['image_size']
            if not isinstance(size, int) or size <= 0:
                self.errors.append("image_size must be positive integer")
            elif size % 32 != 0:
                self.warnings.append("image_size should be multiple of 32 for optimal performance")
            elif size > 512:
                self.warnings.append("image_size > 512 may require significant GPU memory")
        
        # Split ratios
        for split in ['test_split_ratio', 'validation_split_ratio']:
            if split in dataset:
                ratio = dataset[split]
                if not isinstance(ratio, (int, float)) or ratio <= 0 or ratio >= 1:
                    self.errors.append(f"{split} must be between 0 and 1")
    
    def _validate_model_config(self, models: Dict[str, Any]):
        """Validate model configuration parameters."""
        if not models:
            return
        
        # Validate encoder config
        if 'encoder' in models:
            encoder = models['encoder']
            self._validate_encoder_config(encoder)
        
        # Validate palatenet config
        if 'palatenet' in models:
            palatenet = models['palatenet']
            self._validate_palatenet_config(palatenet)
    
    def _validate_encoder_config(self, encoder: Dict[str, Any]):
        """Validate encoder-specific parameters."""
        # Required parameters
        required = ['latent_dim', 'batch_size', 'epochs', 'learning_rate']
        for param in required:
            if param not in encoder:
                self.errors.append(f"Required encoder parameter '{param}' missing")
        
        # Value validation
        if 'latent_dim' in encoder:
            dim = encoder['latent_dim']
            if not isinstance(dim, int) or dim <= 0:
                self.errors.append("encoder.latent_dim must be positive integer")
            elif dim < 64:
                self.warnings.append("encoder.latent_dim < 64 may limit representation quality")
            elif dim > 1024:
                self.warnings.append("encoder.latent_dim > 1024 may require significant resources")
        
        if 'batch_size' in encoder:
            batch = encoder['batch_size']
            if not isinstance(batch, int) or batch <= 0:
                self.errors.append("encoder.batch_size must be positive integer")
            elif batch > 256:
                self.warnings.append("encoder.batch_size > 256 may cause memory issues")
        
        if 'epochs' in encoder:
            epochs = encoder['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                self.errors.append("encoder.epochs must be positive integer")
            elif epochs > 200:
                self.warnings.append("encoder.epochs > 200 may lead to overfitting")
        
        if 'learning_rate' in encoder:
            lr = encoder['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("encoder.learning_rate must be positive number")
            elif lr > 0.1:
                self.warnings.append("encoder.learning_rate > 0.1 may cause training instability")
            elif lr < 0.00001:
                self.warnings.append("encoder.learning_rate < 0.00001 may be too slow")
        
        if 'temperature' in encoder:
            temp = encoder['temperature']
            if not isinstance(temp, (int, float)) or temp <= 0:
                self.errors.append("encoder.temperature must be positive number")
            elif temp > 1.0:
                self.warnings.append("encoder.temperature > 1.0 may reduce contrastive learning effectiveness")
    
    def _validate_palatenet_config(self, palatenet: Dict[str, Any]):
        """Validate palatenet-specific parameters."""
        # Required parameters
        required = ['hidden_dim', 'num_layers', 'batch_size', 'epochs', 'learning_rate']
        for param in required:
            if param not in palatenet:
                self.errors.append(f"Required palatenet parameter '{param}' missing")
        
        # Value validation
        if 'hidden_dim' in palatenet:
            dim = palatenet['hidden_dim']
            if not isinstance(dim, int) or dim <= 0:
                self.errors.append("palatenet.hidden_dim must be positive integer")
            elif dim < 16:
                self.warnings.append("palatenet.hidden_dim < 16 may limit model capacity")
            elif dim > 512:
                self.warnings.append("palatenet.hidden_dim > 512 may require significant resources")
        
        if 'num_layers' in palatenet:
            layers = palatenet['num_layers']
            if not isinstance(layers, int) or layers <= 0:
                self.errors.append("palatenet.num_layers must be positive integer")
            elif layers > 6:
                self.warnings.append("palatenet.num_layers > 6 may cause overfitting or vanishing gradients")
        
        if 'dropout' in palatenet:
            dropout = palatenet['dropout']
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout >= 1:
                self.errors.append("palatenet.dropout must be between 0 and 1")
        
        if 'aggregation' in palatenet:
            agg = palatenet['aggregation']
            valid_agg = ['mean', 'max', 'sum', 'attention']
            if agg not in valid_agg:
                self.errors.append(f"palatenet.aggregation must be one of {valid_agg}")
    
    def _validate_training_config(self, training: Dict[str, Any]):
        """Validate training configuration parameters."""
        if not training:
            return
        
        if 'device' in training:
            device = training['device']
            valid_devices = ['auto', 'cpu', 'cuda', 'mps']
            if device not in valid_devices:
                self.errors.append(f"training.device must be one of {valid_devices}")
        
        if 'max_grad_norm' in training:
            norm = training['max_grad_norm']
            if not isinstance(norm, (int, float)) or norm <= 0:
                self.errors.append("training.max_grad_norm must be positive number")
        
        if 'warmup_steps' in training:
            steps = training['warmup_steps']
            if not isinstance(steps, int) or steps < 0:
                self.errors.append("training.warmup_steps must be non-negative integer")
    
    def _validate_fusion_config(self, fusion: Dict[str, Any]):
        """Validate fusion configuration parameters."""
        if not fusion:
            return
        
        if 'alpha_values' in fusion:
            alphas = fusion['alpha_values']
            if not isinstance(alphas, list) or len(alphas) == 0:
                self.errors.append("fusion.alpha_values must be non-empty list")
            else:
                for alpha in alphas:
                    if not isinstance(alpha, (int, float)) or alpha < 0 or alpha > 1:
                        self.errors.append("All fusion.alpha_values must be between 0 and 1")
                        break
        
        if 'num_samples_per_alpha' in fusion:
            samples = fusion['num_samples_per_alpha']
            if not isinstance(samples, int) or samples <= 0:
                self.errors.append("fusion.num_samples_per_alpha must be positive integer")
            elif samples > 100:
                self.warnings.append("fusion.num_samples_per_alpha > 100 may take significant time")
    
    def _validate_resource_limits(self, resources: Dict[str, Any]):
        """Validate resource limit parameters."""
        if not resources:
            return
        
        if 'max_memory_gb' in resources:
            mem = resources['max_memory_gb']
            if not isinstance(mem, (int, float)) or mem <= 0:
                self.errors.append("resources.max_memory_gb must be positive number")
            elif mem < 2:
                self.warnings.append("resources.max_memory_gb < 2 may be insufficient")
        
        if 'max_disk_gb' in resources:
            disk = resources['max_disk_gb']
            if not isinstance(disk, (int, float)) or disk <= 0:
                self.errors.append("resources.max_disk_gb must be positive number")
            elif disk < 1:
                self.warnings.append("resources.max_disk_gb < 1 may be insufficient")
    
    def _check_parameter_compatibility(self, config: Dict[str, Any]):
        """Check for incompatible parameter combinations."""
        # Get relevant sections
        dataset = config.get('dataset', {})
        models = config.get('models', {})
        training = config.get('training', {})
        
        # Check encoder batch size vs dataset size
        recipes = dataset.get('target_recipes_per_cuisine', 0)
        encoder_batch = models.get('encoder', {}).get('batch_size', 32)
        
        if recipes > 0 and encoder_batch > recipes * 0.8:  # 80% goes to training
            self.warnings.append(
                "encoder.batch_size is very large compared to training set size"
            )
        
        # Check CPU training with large models
        device = training.get('device', 'auto')
        encoder_dim = models.get('encoder', {}).get('latent_dim', 256)
        palatenet_dim = models.get('palatenet', {}).get('hidden_dim', 64)
        
        if device == 'cpu' and (encoder_dim > 512 or palatenet_dim > 128):
            self.warnings.append(
                "Large model dimensions with CPU training may be very slow"
            )
        
        # Check mixed precision with CPU
        mixed_precision = training.get('mixed_precision', False)
        if device == 'cpu' and mixed_precision:
            self.warnings.append(
                "Mixed precision training not supported on CPU, will be disabled"
            )
    
    def _estimate_resources(self, config: Dict[str, Any]):
        """Estimate resource requirements based on configuration."""
        dataset = config.get('dataset', {})
        models = config.get('models', {})
        
        # Estimate memory usage
        recipes = dataset.get('target_recipes_per_cuisine', 5000)
        cuisines = len(dataset.get('target_cuisines', ['japanese', 'italian']))
        encoder_batch = models.get('encoder', {}).get('batch_size', 32)
        encoder_dim = models.get('encoder', {}).get('latent_dim', 256)
        image_size = dataset.get('image_size', 256)
        
        # Rough memory estimation (in GB)
        data_memory = (recipes * cuisines * image_size * image_size * 3) / (1024**3)
        model_memory = (encoder_dim * encoder_batch * 4) / (1024**3)
        estimated_memory = data_memory + model_memory + 2  # Base overhead
        
        self.info.append(f"Estimated memory usage: {estimated_memory:.1f} GB")
        
        # Estimate training time
        encoder_epochs = models.get('encoder', {}).get('epochs', 10)
        palatenet_epochs = models.get('palatenet', {}).get('epochs', 20)
        
        # Very rough time estimation (minutes)
        time_per_recipe = 0.001  # seconds per recipe per epoch
        encoder_time = recipes * cuisines * encoder_epochs * time_per_recipe / 60
        palatenet_time = recipes * cuisines * palatenet_epochs * time_per_recipe / 60
        estimated_time = encoder_time + palatenet_time
        
        self.info.append(f"Estimated training time: {estimated_time:.0f} minutes")
        
        # Check against resource limits
        max_memory = config.get('resources', {}).get('max_memory_gb', 8)
        if estimated_memory > max_memory:
            self.warnings.append(
                f"Estimated memory usage ({estimated_memory:.1f} GB) exceeds limit ({max_memory} GB)"
            )
    
    def print_results(self):
        """Print validation results."""
        print("=" * 60)
        print("CONFIGURATION VALIDATION RESULTS")
        print("=" * 60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.info:
            print(f"\n📊 ESTIMATES:")
            for info in self.info:
                print(f"  • {info}")
        
        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid with no issues!")
        elif not self.errors:
            print("\n✅ Configuration is valid (warnings can be ignored)")
        else:
            print("\n❌ Configuration has errors that must be fixed")
        
        print("=" * 60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Validate fusion cuisine configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_config.py config.yaml
  python scripts/validate_config.py configs/production.yaml
  python scripts/validate_config.py configs/quick_prototype.yaml
        """
    )
    
    parser.add_argument(
        'config_file',
        help='Path to configuration file to validate'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show errors, suppress warnings and info'
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    validator = ConfigValidator()
    is_valid = validator.validate(args.config_file)
    
    # Print results (unless quiet mode with no errors)
    if not (args.quiet and not validator.errors):
        if args.quiet:
            # Only show errors in quiet mode
            if validator.errors:
                print("ERRORS:")
                for error in validator.errors:
                    print(f"  • {error}")
        else:
            validator.print_results()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()