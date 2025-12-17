#!/usr/bin/env python3
"""
Resource Estimation Script for Fusion Cuisine POC

This script estimates memory usage, training time, and disk space
requirements based on configuration parameters.

Usage:
    python scripts/estimate_resources.py config.yaml
    python scripts/estimate_resources.py configs/production.yaml --detailed
"""

import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import json


class ResourceEstimator:
    """Estimates resource requirements for fusion cuisine training."""
    
    def __init__(self):
        # Hardware-specific multipliers
        self.device_multipliers = {
            'cpu': {'speed': 0.1, 'memory': 1.0},
            'cuda': {'speed': 1.0, 'memory': 1.2},
            'mps': {'speed': 0.8, 'memory': 1.1},
            'auto': {'speed': 1.0, 'memory': 1.2}  # Assume GPU
        }
        
        # Base computational costs (in arbitrary units)
        self.base_costs = {
            'recipe_processing': 0.001,  # seconds per recipe
            'image_processing': 0.01,    # seconds per image
            'encoder_training': 0.1,     # seconds per batch
            'palatenet_training': 0.05,  # seconds per batch
            'fusion_generation': 1.0,    # seconds per fusion recipe
        }
    
    def estimate(self, config_path: str) -> Dict[str, Any]:
        """Estimate resources for given configuration."""
        config = self._load_config(config_path)
        
        estimates = {
            'memory': self._estimate_memory(config),
            'training_time': self._estimate_training_time(config),
            'disk_space': self._estimate_disk_space(config),
            'hardware_requirements': self._estimate_hardware_requirements(config),
            'cost_breakdown': self._estimate_cost_breakdown(config)
        }
        
        return estimates
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _estimate_memory(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory usage in GB."""
        dataset = config.get('dataset', {})
        models = config.get('models', {})
        training = config.get('training', {})
        
        # Dataset parameters
        recipes = dataset.get('target_recipes_per_cuisine', 5000)
        cuisines = len(dataset.get('target_cuisines', ['japanese', 'italian']))
        image_size = dataset.get('image_size', 256)
        text_length = config.get('preprocessing', {}).get('text_max_length', 512)
        
        # Model parameters
        encoder_batch = models.get('encoder', {}).get('batch_size', 32)
        palatenet_batch = models.get('palatenet', {}).get('batch_size', 64)
        latent_dim = models.get('encoder', {}).get('latent_dim', 256)
        hidden_dim = models.get('palatenet', {}).get('hidden_dim', 64)
        
        # Memory calculations (in GB)
        total_recipes = recipes * cuisines
        
        # Data memory
        image_memory = (total_recipes * image_size * image_size * 3 * 4) / (1024**3)  # Float32
        text_memory = (total_recipes * text_length * 4) / (1024**3)  # Embeddings
        graph_memory = 0.1  # FlavorGraph is relatively small
        
        # Model memory
        encoder_params = latent_dim * 2048  # Rough estimate for CLIP-mini
        palatenet_params = hidden_dim * hidden_dim * models.get('palatenet', {}).get('num_layers', 2)
        model_memory = (encoder_params + palatenet_params) * 4 / (1024**3)  # Float32
        
        # Training memory (gradients, optimizer states)
        training_overhead = model_memory * 3  # Gradients + optimizer states
        
        # Batch memory
        encoder_batch_memory = (encoder_batch * image_size * image_size * 3 * 4) / (1024**3)
        palatenet_batch_memory = (palatenet_batch * latent_dim * 4) / (1024**3)
        
        # System overhead
        system_overhead = 1.0  # Base system requirements
        
        # Mixed precision reduction
        mixed_precision = training.get('mixed_precision', False)
        precision_factor = 0.6 if mixed_precision else 1.0
        
        return {
            'data_memory': image_memory + text_memory + graph_memory,
            'model_memory': model_memory,
            'training_overhead': training_overhead * precision_factor,
            'batch_memory': max(encoder_batch_memory, palatenet_batch_memory),
            'system_overhead': system_overhead,
            'total_memory': (image_memory + text_memory + graph_memory + 
                           model_memory + training_overhead * precision_factor +
                           max(encoder_batch_memory, palatenet_batch_memory) + 
                           system_overhead)
        }
    
    def _estimate_training_time(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate training time in minutes."""
        dataset = config.get('dataset', {})
        models = config.get('models', {})
        training = config.get('training', {})
        fusion = config.get('fusion', {})
        
        # Dataset parameters
        recipes = dataset.get('target_recipes_per_cuisine', 5000)
        cuisines = len(dataset.get('target_cuisines', ['japanese', 'italian']))
        total_recipes = recipes * cuisines
        
        # Training parameters
        encoder_epochs = models.get('encoder', {}).get('epochs', 10)
        palatenet_epochs = models.get('palatenet', {}).get('epochs', 20)
        encoder_batch = models.get('encoder', {}).get('batch_size', 32)
        palatenet_batch = models.get('palatenet', {}).get('batch_size', 64)
        
        # Fusion parameters
        alpha_values = len(fusion.get('alpha_values', [0.3, 0.5, 0.7]))
        samples_per_alpha = fusion.get('num_samples_per_alpha', 10)
        
        # Device multiplier
        device = training.get('device', 'auto')
        speed_multiplier = self.device_multipliers[device]['speed']
        
        # Time calculations (in minutes)
        # Data preprocessing
        data_prep_time = (total_recipes * self.base_costs['recipe_processing'] + 
                         total_recipes * self.base_costs['image_processing']) / 60
        
        # Encoder training
        encoder_batches = total_recipes / encoder_batch
        encoder_time = (encoder_epochs * encoder_batches * 
                       self.base_costs['encoder_training']) / 60 / speed_multiplier
        
        # PalateNet training
        palatenet_batches = total_recipes / palatenet_batch
        palatenet_time = (palatenet_epochs * palatenet_batches * 
                         self.base_costs['palatenet_training']) / 60 / speed_multiplier
        
        # Fusion generation
        total_fusion_recipes = alpha_values * samples_per_alpha
        fusion_time = (total_fusion_recipes * self.base_costs['fusion_generation']) / 60
        
        # Evaluation
        eval_time = 5  # Rough estimate for evaluation metrics
        
        return {
            'data_preprocessing': data_prep_time,
            'encoder_training': encoder_time,
            'palatenet_training': palatenet_time,
            'fusion_generation': fusion_time,
            'evaluation': eval_time,
            'total_time': data_prep_time + encoder_time + palatenet_time + fusion_time + eval_time
        }
    
    def _estimate_disk_space(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate disk space requirements in GB."""
        dataset = config.get('dataset', {})
        models = config.get('models', {})
        fusion = config.get('fusion', {})
        
        # Dataset parameters
        recipes = dataset.get('target_recipes_per_cuisine', 5000)
        cuisines = len(dataset.get('target_cuisines', ['japanese', 'italian']))
        image_size = dataset.get('image_size', 256)
        total_recipes = recipes * cuisines
        
        # Storage calculations (in GB)
        # Raw images (compressed)
        image_storage = (total_recipes * image_size * image_size * 3 * 0.1) / (1024**3)  # JPEG compression
        
        # Processed data
        processed_data = 0.5  # Embeddings, graphs, etc.
        
        # Model checkpoints
        latent_dim = models.get('encoder', {}).get('latent_dim', 256)
        hidden_dim = models.get('palatenet', {}).get('hidden_dim', 64)
        model_size = (latent_dim * 2048 + hidden_dim * hidden_dim * 2) * 4 / (1024**3)
        model_checkpoints = model_size * 5  # Keep multiple checkpoints
        
        # Output recipes and images
        alpha_values = len(fusion.get('alpha_values', [0.3, 0.5, 0.7]))
        samples_per_alpha = fusion.get('num_samples_per_alpha', 10)
        output_storage = alpha_values * samples_per_alpha * 0.001  # Small text files
        
        # Logs and temporary files
        logs_temp = 0.1
        
        return {
            'input_images': image_storage,
            'processed_data': processed_data,
            'model_checkpoints': model_checkpoints,
            'output_files': output_storage,
            'logs_temp': logs_temp,
            'total_disk': image_storage + processed_data + model_checkpoints + output_storage + logs_temp
        }
    
    def _estimate_hardware_requirements(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate minimum hardware requirements."""
        memory_est = self._estimate_memory(config)
        time_est = self._estimate_training_time(config)
        disk_est = self._estimate_disk_space(config)
        
        total_memory = memory_est['total_memory']
        total_time = time_est['total_time']
        total_disk = disk_est['total_disk']
        
        # Categorize requirements
        if total_memory <= 4 and total_time <= 60:
            category = "Budget/Laptop"
            recommendations = {
                'cpu': "4+ cores, 2.5+ GHz",
                'memory': "8GB+ RAM",
                'gpu': "Optional (GTX 1660 or better)",
                'storage': "10GB+ SSD space"
            }
        elif total_memory <= 8 and total_time <= 180:
            category = "Mid-range Desktop"
            recommendations = {
                'cpu': "6+ cores, 3.0+ GHz",
                'memory': "16GB+ RAM",
                'gpu': "RTX 3060 or better",
                'storage': "20GB+ SSD space"
            }
        elif total_memory <= 16 and total_time <= 480:
            category = "High-end Desktop"
            recommendations = {
                'cpu': "8+ cores, 3.5+ GHz",
                'memory': "32GB+ RAM",
                'gpu': "RTX 3080/4070 or better",
                'storage': "50GB+ NVMe SSD"
            }
        else:
            category = "Workstation/Server"
            recommendations = {
                'cpu': "12+ cores, 3.5+ GHz",
                'memory': "64GB+ RAM",
                'gpu': "RTX 4090/A6000 or better",
                'storage': "100GB+ NVMe SSD"
            }
        
        return {
            'category': category,
            'estimated_memory_gb': total_memory,
            'estimated_time_hours': total_time / 60,
            'estimated_disk_gb': total_disk,
            'recommendations': recommendations
        }
    
    def _estimate_cost_breakdown(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Break down computational costs by component."""
        time_est = self._estimate_training_time(config)
        total_time = time_est['total_time']
        
        # Percentage breakdown
        breakdown = {}
        for component, time_val in time_est.items():
            if component != 'total_time':
                breakdown[component] = (time_val / total_time) * 100 if total_time > 0 else 0
        
        return breakdown


def format_time(minutes: float) -> str:
    """Format time in human-readable format."""
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    elif minutes < 1440:
        return f"{minutes/60:.1f} hours"
    else:
        return f"{minutes/1440:.1f} days"


def format_size(gb: float) -> str:
    """Format size in human-readable format."""
    if gb < 1:
        return f"{gb*1024:.0f} MB"
    elif gb < 1024:
        return f"{gb:.1f} GB"
    else:
        return f"{gb/1024:.1f} TB"


def print_estimates(estimates: Dict[str, Any], detailed: bool = False):
    """Print resource estimates in a formatted way."""
    print("=" * 70)
    print("RESOURCE ESTIMATION REPORT")
    print("=" * 70)
    
    # Memory breakdown
    memory = estimates['memory']
    print(f"\n📊 MEMORY REQUIREMENTS:")
    print(f"  Total Estimated Memory: {format_size(memory['total_memory'])}")
    
    if detailed:
        print(f"    • Data Memory:        {format_size(memory['data_memory'])}")
        print(f"    • Model Memory:       {format_size(memory['model_memory'])}")
        print(f"    • Training Overhead:  {format_size(memory['training_overhead'])}")
        print(f"    • Batch Memory:       {format_size(memory['batch_memory'])}")
        print(f"    • System Overhead:    {format_size(memory['system_overhead'])}")
    
    # Training time breakdown
    time_est = estimates['training_time']
    print(f"\n⏱️  TRAINING TIME:")
    print(f"  Total Estimated Time: {format_time(time_est['total_time'])}")
    
    if detailed:
        print(f"    • Data Preprocessing: {format_time(time_est['data_preprocessing'])}")
        print(f"    • Encoder Training:   {format_time(time_est['encoder_training'])}")
        print(f"    • PalateNet Training: {format_time(time_est['palatenet_training'])}")
        print(f"    • Fusion Generation:  {format_time(time_est['fusion_generation'])}")
        print(f"    • Evaluation:         {format_time(time_est['evaluation'])}")
    
    # Disk space breakdown
    disk = estimates['disk_space']
    print(f"\n💾 DISK SPACE:")
    print(f"  Total Estimated Space: {format_size(disk['total_disk'])}")
    
    if detailed:
        print(f"    • Input Images:       {format_size(disk['input_images'])}")
        print(f"    • Processed Data:     {format_size(disk['processed_data'])}")
        print(f"    • Model Checkpoints:  {format_size(disk['model_checkpoints'])}")
        print(f"    • Output Files:       {format_size(disk['output_files'])}")
        print(f"    • Logs & Temp:        {format_size(disk['logs_temp'])}")
    
    # Hardware requirements
    hw = estimates['hardware_requirements']
    print(f"\n🖥️  HARDWARE CATEGORY: {hw['category']}")
    print(f"  Recommendations:")
    for component, spec in hw['recommendations'].items():
        print(f"    • {component.upper()}: {spec}")
    
    # Cost breakdown
    if detailed:
        costs = estimates['cost_breakdown']
        print(f"\n📈 COMPUTATION BREAKDOWN:")
        for component, percentage in costs.items():
            component_name = component.replace('_', ' ').title()
            print(f"    • {component_name}: {percentage:.1f}%")
    
    print("=" * 70)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Estimate resource requirements for fusion cuisine training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/estimate_resources.py config.yaml
  python scripts/estimate_resources.py configs/production.yaml --detailed
  python scripts/estimate_resources.py configs/quick_prototype.yaml --json
        """
    )
    
    parser.add_argument(
        'config_file',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed breakdown of estimates'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output estimates in JSON format'
    )
    
    args = parser.parse_args()
    
    try:
        # Estimate resources
        estimator = ResourceEstimator()
        estimates = estimator.estimate(args.config_file)
        
        if args.json:
            # JSON output for programmatic use
            print(json.dumps(estimates, indent=2))
        else:
            # Human-readable output
            print_estimates(estimates, args.detailed)
            
    except Exception as e:
        print(f"Error estimating resources: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()