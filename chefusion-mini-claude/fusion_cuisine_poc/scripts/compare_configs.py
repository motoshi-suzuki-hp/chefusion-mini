#!/usr/bin/env python3
"""
Configuration Comparison Tool for Fusion Cuisine POC

This script compares two configuration files and highlights differences,
estimates performance impacts, and suggests optimization strategies.

Usage:
    python scripts/compare_configs.py config1.yaml config2.yaml
    python scripts/compare_configs.py configs/quick_prototype.yaml configs/production.yaml --impact
"""

import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
from dataclasses import dataclass


@dataclass
class ConfigDifference:
    """Represents a difference between two configurations."""
    path: str
    value1: Any
    value2: Any
    impact_level: str  # 'high', 'medium', 'low'
    description: str


class ConfigComparator:
    """Compares fusion cuisine configuration files."""
    
    def __init__(self):
        # Define impact levels for different parameters
        self.impact_levels = {
            # High impact parameters (major performance/quality changes)
            'dataset.target_recipes_per_cuisine': 'high',
            'models.encoder.latent_dim': 'high',
            'models.encoder.epochs': 'high',
            'models.palatenet.hidden_dim': 'high',
            'models.palatenet.epochs': 'high',
            'models.palatenet.num_layers': 'high',
            'dataset.target_cuisines': 'high',
            
            # Medium impact parameters (noticeable changes)
            'models.encoder.batch_size': 'medium',
            'models.palatenet.batch_size': 'medium',
            'models.encoder.learning_rate': 'medium',
            'models.palatenet.learning_rate': 'medium',
            'dataset.image_size': 'medium',
            'models.palatenet.dropout': 'medium',
            'fusion.alpha_values': 'medium',
            'fusion.num_samples_per_alpha': 'medium',
            
            # Low impact parameters (minor tweaks)
            'training.max_grad_norm': 'low',
            'training.warmup_steps': 'low',
            'models.encoder.temperature': 'low',
            'models.encoder.projection_dim': 'low',
            'preprocessing.text_max_length': 'low',
            'environment.num_workers': 'low',
            'training.save_steps': 'low',
            'training.eval_steps': 'low',
        }
        
        # Parameter descriptions for better understanding
        self.descriptions = {
            'dataset.target_recipes_per_cuisine': 'Dataset size per cuisine - affects model quality and training time',
            'models.encoder.latent_dim': 'Embedding dimension - affects representation quality and memory usage',
            'models.encoder.epochs': 'Encoder training duration - affects representation quality',
            'models.palatenet.hidden_dim': 'Model capacity - affects prediction accuracy and memory usage',
            'models.palatenet.epochs': 'PalateNet training duration - affects rating prediction quality',
            'models.palatenet.num_layers': 'Model depth - affects capacity and overfitting risk',
            'dataset.target_cuisines': 'Number of cuisines - affects complexity and training time',
            'models.encoder.batch_size': 'Training batch size - affects memory usage and convergence',
            'models.palatenet.batch_size': 'Training batch size - affects memory usage and stability',
            'models.encoder.learning_rate': 'Learning speed - affects convergence and stability',
            'models.palatenet.learning_rate': 'Learning speed - affects convergence and stability',
            'dataset.image_size': 'Image resolution - affects quality and computational cost',
            'models.palatenet.dropout': 'Regularization strength - affects overfitting',
            'fusion.alpha_values': 'Cultural blending ratios - affects fusion diversity',
            'fusion.num_samples_per_alpha': 'Number of fusion recipes per alpha - affects output diversity',
        }
    
    def compare(self, config1_path: str, config2_path: str) -> List[ConfigDifference]:
        """Compare two configuration files."""
        config1 = self._load_config(config1_path)
        config2 = self._load_config(config2_path)
        
        differences = []
        
        # Find all differences recursively
        self._find_differences(config1, config2, "", differences)
        
        return differences
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _find_differences(self, dict1: Dict[str, Any], dict2: Dict[str, Any], 
                         path: str, differences: List[ConfigDifference]):
        """Recursively find differences between two dictionaries."""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                # Key only in dict2
                differences.append(ConfigDifference(
                    path=current_path,
                    value1=None,
                    value2=dict2[key],
                    impact_level=self._get_impact_level(current_path),
                    description=self._get_description(current_path)
                ))
            elif key not in dict2:
                # Key only in dict1
                differences.append(ConfigDifference(
                    path=current_path,
                    value1=dict1[key],
                    value2=None,
                    impact_level=self._get_impact_level(current_path),
                    description=self._get_description(current_path)
                ))
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # Both are dictionaries, recurse
                self._find_differences(dict1[key], dict2[key], current_path, differences)
            elif dict1[key] != dict2[key]:
                # Values are different
                differences.append(ConfigDifference(
                    path=current_path,
                    value1=dict1[key],
                    value2=dict2[key],
                    impact_level=self._get_impact_level(current_path),
                    description=self._get_description(current_path)
                ))
    
    def _get_impact_level(self, path: str) -> str:
        """Get impact level for a parameter path."""
        return self.impact_levels.get(path, 'low')
    
    def _get_description(self, path: str) -> str:
        """Get description for a parameter path."""
        return self.descriptions.get(path, f"Configuration parameter: {path}")
    
    def estimate_performance_impact(self, differences: List[ConfigDifference]) -> Dict[str, Any]:
        """Estimate the performance impact of configuration differences."""
        impact_summary = {
            'high_impact_changes': 0,
            'medium_impact_changes': 0,
            'low_impact_changes': 0,
            'estimated_impact': {},
            'recommendations': []
        }
        
        for diff in differences:
            if diff.impact_level == 'high':
                impact_summary['high_impact_changes'] += 1
            elif diff.impact_level == 'medium':
                impact_summary['medium_impact_changes'] += 1
            else:
                impact_summary['low_impact_changes'] += 1
        
        # Estimate specific impacts
        impact_summary['estimated_impact'] = self._estimate_specific_impacts(differences)
        
        # Generate recommendations
        impact_summary['recommendations'] = self._generate_recommendations(differences)
        
        return impact_summary
    
    def _estimate_specific_impacts(self, differences: List[ConfigDifference]) -> Dict[str, str]:
        """Estimate specific performance impacts."""
        impacts = {}
        
        for diff in differences:
            if 'target_recipes_per_cuisine' in diff.path:
                if diff.value2 and diff.value1:
                    ratio = diff.value2 / diff.value1
                    if ratio > 2:
                        impacts['training_time'] = f"Significantly longer (~{ratio:.1f}x)"
                        impacts['quality'] = "Likely better due to more data"
                    elif ratio < 0.5:
                        impacts['training_time'] = f"Significantly shorter (~{1/ratio:.1f}x faster)"
                        impacts['quality'] = "May be lower due to less data"
            
            elif 'latent_dim' in diff.path:
                if diff.value2 and diff.value1:
                    ratio = diff.value2 / diff.value1
                    if ratio > 1.5:
                        impacts['memory_usage'] = f"Higher (~{ratio:.1f}x more memory)"
                        impacts['representation_quality'] = "Likely better"
                    elif ratio < 0.7:
                        impacts['memory_usage'] = f"Lower (~{1/ratio:.1f}x less memory)"
                        impacts['representation_quality'] = "May be lower"
            
            elif 'epochs' in diff.path:
                if diff.value2 and diff.value1:
                    ratio = diff.value2 / diff.value1
                    if ratio > 2:
                        impacts['training_time'] = f"Much longer (~{ratio:.1f}x)"
                        impacts['convergence'] = "Better convergence expected"
                    elif ratio < 0.5:
                        impacts['training_time'] = f"Much shorter (~{1/ratio:.1f}x)"
                        impacts['convergence'] = "May not converge fully"
            
            elif 'batch_size' in diff.path:
                if diff.value2 and diff.value1:
                    ratio = diff.value2 / diff.value1
                    if ratio > 2:
                        impacts['memory_usage'] = f"Higher (~{ratio:.1f}x more memory)"
                        impacts['training_stability'] = "More stable gradients"
                    elif ratio < 0.5:
                        impacts['memory_usage'] = f"Lower (~{1/ratio:.1f}x less memory)"
                        impacts['training_stability'] = "Less stable gradients"
        
        return impacts
    
    def _generate_recommendations(self, differences: List[ConfigDifference]) -> List[str]:
        """Generate recommendations based on differences."""
        recommendations = []
        
        high_impact_diffs = [d for d in differences if d.impact_level == 'high']
        
        if high_impact_diffs:
            recommendations.append("⚠️  High-impact changes detected - test thoroughly before production")
        
        # Check for common issues
        memory_intensive = False
        time_intensive = False
        
        for diff in differences:
            if ('latent_dim' in diff.path or 'hidden_dim' in diff.path) and diff.value2 and diff.value1:
                if diff.value2 > diff.value1 * 2:
                    memory_intensive = True
            
            if ('epochs' in diff.path or 'target_recipes_per_cuisine' in diff.path) and diff.value2 and diff.value1:
                if diff.value2 > diff.value1 * 3:
                    time_intensive = True
        
        if memory_intensive:
            recommendations.append("💾 Consider enabling mixed_precision and gradient_checkpointing for memory efficiency")
        
        if time_intensive:
            recommendations.append("⏱️  Long training expected - consider starting with smaller scale for testing")
        
        # Check for learning rate changes
        lr_changes = [d for d in differences if 'learning_rate' in d.path]
        if lr_changes:
            for diff in lr_changes:
                if diff.value2 and diff.value1 and diff.value2 > diff.value1 * 5:
                    recommendations.append("🚀 High learning rate detected - monitor for training instability")
                elif diff.value2 and diff.value1 and diff.value2 < diff.value1 * 0.1:
                    recommendations.append("🐌 Very low learning rate - training may be very slow")
        
        return recommendations


def format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "None"
    elif isinstance(value, list):
        if len(value) > 5:
            return f"[{', '.join(map(str, value[:3]))}, ... ({len(value)} items)]"
        else:
            return str(value)
    else:
        return str(value)


def print_comparison(differences: List[ConfigDifference], config1_name: str, config2_name: str, 
                    show_impact: bool = False):
    """Print configuration comparison results."""
    print("=" * 80)
    print(f"CONFIGURATION COMPARISON: {config1_name} vs {config2_name}")
    print("=" * 80)
    
    if not differences:
        print("\n✅ No differences found - configurations are identical!")
        print("=" * 80)
        return
    
    # Group by impact level
    high_impact = [d for d in differences if d.impact_level == 'high']
    medium_impact = [d for d in differences if d.impact_level == 'medium']
    low_impact = [d for d in differences if d.impact_level == 'low']
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total differences: {len(differences)}")
    print(f"  High impact:       {len(high_impact)}")
    print(f"  Medium impact:     {len(medium_impact)}")
    print(f"  Low impact:        {len(low_impact)}")
    
    # Show high impact differences first
    if high_impact:
        print(f"\n🔴 HIGH IMPACT DIFFERENCES ({len(high_impact)}):")
        for diff in high_impact:
            print(f"\n  Parameter: {diff.path}")
            print(f"  {config1_name}: {format_value(diff.value1)}")
            print(f"  {config2_name}: {format_value(diff.value2)}")
            print(f"  Impact: {diff.description}")
    
    # Show medium impact differences
    if medium_impact:
        print(f"\n🟡 MEDIUM IMPACT DIFFERENCES ({len(medium_impact)}):")
        for diff in medium_impact:
            print(f"\n  Parameter: {diff.path}")
            print(f"  {config1_name}: {format_value(diff.value1)}")
            print(f"  {config2_name}: {format_value(diff.value2)}")
            print(f"  Impact: {diff.description}")
    
    # Show low impact differences (condensed)
    if low_impact:
        print(f"\n🟢 LOW IMPACT DIFFERENCES ({len(low_impact)}):")
        for diff in low_impact:
            print(f"  • {diff.path}: {format_value(diff.value1)} → {format_value(diff.value2)}")
    
    # Performance impact analysis
    if show_impact:
        comparator = ConfigComparator()
        impact_analysis = comparator.estimate_performance_impact(differences)
        
        print(f"\n🎯 PERFORMANCE IMPACT ANALYSIS:")
        
        if impact_analysis['estimated_impact']:
            print(f"  Estimated changes:")
            for category, impact in impact_analysis['estimated_impact'].items():
                print(f"    • {category.replace('_', ' ').title()}: {impact}")
        
        if impact_analysis['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in impact_analysis['recommendations']:
                print(f"  {rec}")
    
    print("=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare fusion cuisine configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_configs.py config1.yaml config2.yaml
  python scripts/compare_configs.py configs/quick_prototype.yaml configs/production.yaml --impact
  python scripts/compare_configs.py old_config.yaml new_config.yaml --json
        """
    )
    
    parser.add_argument(
        'config1',
        help='Path to first configuration file'
    )
    
    parser.add_argument(
        'config2',
        help='Path to second configuration file'
    )
    
    parser.add_argument(
        '--impact', '-i',
        action='store_true',
        help='Show performance impact analysis'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output differences in JSON format'
    )
    
    parser.add_argument(
        '--high-only',
        action='store_true',
        help='Only show high-impact differences'
    )
    
    args = parser.parse_args()
    
    try:
        # Compare configurations
        comparator = ConfigComparator()
        differences = comparator.compare(args.config1, args.config2)
        
        # Filter for high-impact only if requested
        if args.high_only:
            differences = [d for d in differences if d.impact_level == 'high']
        
        if args.json:
            # JSON output for programmatic use
            output = {
                'config1': args.config1,
                'config2': args.config2,
                'total_differences': len(differences),
                'differences': [
                    {
                        'path': d.path,
                        'value1': d.value1,
                        'value2': d.value2,
                        'impact_level': d.impact_level,
                        'description': d.description
                    }
                    for d in differences
                ]
            }
            
            if args.impact:
                impact_analysis = comparator.estimate_performance_impact(differences)
                output['impact_analysis'] = impact_analysis
            
            print(json.dumps(output, indent=2, default=str))
        else:
            # Human-readable output
            config1_name = Path(args.config1).name
            config2_name = Path(args.config2).name
            print_comparison(differences, config1_name, config2_name, args.impact)
            
    except Exception as e:
        print(f"Error comparing configurations: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()