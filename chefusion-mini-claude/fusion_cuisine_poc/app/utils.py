"""Utility functions for the fusion cuisine POC."""

import json
import logging
import pickle
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "fusion_cuisine.log", mode="a"),
        ],
    )


def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Path) -> None:
    """Save data to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: Path) -> Any:
    """Load data from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_device() -> torch.device:
    """Get the appropriate device for PyTorch operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resize_image(image: Image.Image, size: int) -> Image.Image:
    """Resize image to specified size while maintaining aspect ratio."""
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return image


def clean_recipe_text(text: str) -> str:
    """Clean and normalize recipe text."""
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())  # Remove extra whitespace
    
    return text


def extract_ingredients(recipe_text: str) -> List[str]:
    """Extract ingredients from recipe text (simple implementation)."""
    # This is a simplified implementation
    # In practice, you'd use NLP techniques or structured data
    ingredients = []
    lines = recipe_text.split(".")
    
    for line in lines:
        line = line.strip().lower()
        if any(word in line for word in ["ingredient", "need", "use", "add"]):
            # Extract potential ingredients (simplified)
            words = line.split()
            ingredients.extend([word for word in words if len(word) > 3])
    
    return list(set(ingredients))


class RecipeDataset(Dataset):
    """Dataset class for recipe data."""
    
    def __init__(
        self,
        recipes: pd.DataFrame,
        image_dir: Optional[Path] = None,
        transform: Optional[Any] = None,
    ):
        self.recipes = recipes
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.recipes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        recipe = self.recipes.iloc[idx]
        
        item = {
            "id": recipe["id"],
            "title": recipe["title"],
            "ingredients": recipe["ingredients"],
            "instructions": recipe["instructions"],
            "cuisine": recipe["cuisine"],
            "rating": recipe.get("rating", 0.0),
        }
        
        # Load image if available
        if self.image_dir and "image_path" in recipe:
            image_path = self.image_dir / recipe["image_path"]
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    item["image"] = image
                except Exception as e:
                    logging.warning(f"Failed to load image {image_path}: {e}")
                    item["image"] = None
            else:
                item["image"] = None
        
        return item


def compute_ingredient_overlap(ingredients1: List[str], ingredients2: List[str]) -> float:
    """Compute ingredient overlap percentage between two ingredient lists."""
    set1 = set(ingredients1)
    set2 = set(ingredients2)
    
    if not set1 and not set2:
        return 1.0
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union) if union else 0.0


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    for metric, value in metrics.items():
        print(f"{metric:<30}: {value:.4f}")
    
    print("="*50 + "\n")


def create_adjacency_matrix(flavor_graph: Dict[str, List[str]]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create adjacency matrix from flavor graph."""
    ingredients = list(flavor_graph.keys())
    ingredient_to_idx = {ingredient: i for i, ingredient in enumerate(ingredients)}
    
    n = len(ingredients)
    adj_matrix = np.zeros((n, n), dtype=np.float32)
    
    for ingredient, connections in flavor_graph.items():
        i = ingredient_to_idx[ingredient]
        for connected_ingredient in connections:
            if connected_ingredient in ingredient_to_idx:
                j = ingredient_to_idx[connected_ingredient]
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # Make symmetric
    
    return adj_matrix, ingredient_to_idx


def split_dataset(
    df: pd.DataFrame, 
    test_ratio: float = 0.1, 
    val_ratio: float = 0.1, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets."""
    # Shuffle the dataset
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n = len(df_shuffled)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    
    test_df = df_shuffled[:test_size]
    val_df = df_shuffled[test_size:test_size + val_size]
    train_df = df_shuffled[test_size + val_size:]
    
    return train_df, val_df, test_df


def normalize_ratings(ratings: pd.Series) -> pd.Series:
    """Normalize ratings to 0-1 range."""
    min_rating = ratings.min()
    max_rating = ratings.max()
    
    if max_rating == min_rating:
        return pd.Series([0.5] * len(ratings))
    
    return (ratings - min_rating) / (max_rating - min_rating)


def compute_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation coefficient."""
    from scipy.stats import spearmanr
    
    correlation, p_value = spearmanr(y_true, y_pred)
    return correlation if not np.isnan(correlation) else 0.0


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cleanup_large_files(directory: Path, max_size_mb: int = 500) -> None:
    """Clean up files larger than max_size_mb in the specified directory."""
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                logging.info(f"Removing large file: {file_path} ({size_mb:.1f}MB)")
                file_path.unlink()


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_model_size(model: torch.nn.Module) -> str:
    """Get the size of a PyTorch model in human-readable format."""
    param_count = sum(p.numel() for p in model.parameters())
    
    if param_count < 1e6:
        return f"{param_count/1e3:.1f}K"
    elif param_count < 1e9:
        return f"{param_count/1e6:.1f}M"
    else:
        return f"{param_count/1e9:.1f}B"


class LogMonitor:
    """
    Real-time log monitoring and analysis system for the fusion cuisine application.
    
    Features:
    - Real-time log file monitoring
    - Error detection and alerting
    - Performance metrics tracking
    - Log analysis and statistics
    - Configurable alert thresholds
    """
    
    def __init__(self, log_file: str = "logs/fusion_cuisine.log", 
                 error_threshold: int = 10, warning_threshold: int = 20):
        """
        Initialize the log monitor.
        
        Args:
            log_file: Path to the log file to monitor
            error_threshold: Number of errors per minute to trigger alert
            warning_threshold: Number of warnings per minute to trigger alert
        """
        self.log_file = Path(log_file)
        self.error_threshold = error_threshold
        self.warning_threshold = warning_threshold
        self.running = False
        self.monitor_thread = None
        
        # Statistics tracking
        self.stats = {
            'total_lines': 0,
            'info_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'critical_count': 0,
            'start_time': None,
            'last_activity': None
        }
        
        # Recent events for rate limiting
        self.recent_errors = deque(maxlen=100)
        self.recent_warnings = deque(maxlen=100)
        
        # Alert handlers
        self.alert_handlers = []
        
        # Performance metrics
        self.performance_metrics = defaultdict(list)
        
        # Create logs directory if it doesn't exist
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Setup monitoring logger
        self.logger = logging.getLogger('log_monitor')
        self.logger.setLevel(logging.INFO)
        
        # Create separate log file for monitoring events
        monitor_log = self.log_file.parent / "monitor.log"
        monitor_handler = logging.FileHandler(monitor_log, mode='a')
        monitor_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(monitor_handler)
    
    def add_alert_handler(self, handler):
        """Add a custom alert handler function."""
        self.alert_handlers.append(handler)
    
    def start_monitoring(self):
        """Start the log monitoring in a separate thread."""
        if self.running:
            self.logger.warning("Log monitoring is already running")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Log monitoring started")
    
    def stop_monitoring(self):
        """Stop the log monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Log monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            # Create log file if it doesn't exist
            if not self.log_file.exists():
                self.log_file.touch()
            
            with open(self.log_file, 'r') as f:
                # Move to end of file for real-time monitoring
                f.seek(0, 2)
                
                while self.running:
                    line = f.readline()
                    if line:
                        self._process_log_line(line.strip())
                    else:
                        time.sleep(0.1)  # Small delay to prevent busy waiting
                        
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
        
    def _process_log_line(self, line: str):
        """Process a single log line."""
        if not line:
            return
        
        self.stats['total_lines'] += 1
        self.stats['last_activity'] = datetime.now()
        
        # Parse log level
        level = self._extract_log_level(line)
        if level:
            self.stats[f'{level.lower()}_count'] += 1
            
            # Track recent errors and warnings for rate limiting
            now = datetime.now()
            if level == 'ERROR':
                self.recent_errors.append(now)
                self._check_error_rate()
            elif level == 'WARNING':
                self.recent_warnings.append(now)
                self._check_warning_rate()
            elif level == 'CRITICAL':
                self._trigger_alert(f"CRITICAL error detected: {line}")
        
        # Extract performance metrics
        self._extract_performance_metrics(line)
        
        # Check for specific patterns
        self._check_patterns(line)
    
    def _extract_log_level(self, line: str) -> Optional[str]:
        """Extract log level from log line."""
        levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        for level in levels:
            if f' - {level} - ' in line:
                return level
        return None
    
    def _extract_performance_metrics(self, line: str):
        """Extract performance metrics from log lines."""
        # Extract timing information
        if 'took' in line.lower() or 'time:' in line.lower():
            try:
                # Look for time patterns like "took 5.2s" or "time: 1.5m"
                import re
                time_pattern = r'(\d+\.?\d*)\s*(s|seconds?|m|minutes?|h|hours?)'
                match = re.search(time_pattern, line.lower())
                if match:
                    value, unit = match.groups()
                    # Convert to seconds
                    multiplier = {'s': 1, 'second': 1, 'seconds': 1, 
                                'm': 60, 'minute': 60, 'minutes': 60,
                                'h': 3600, 'hour': 3600, 'hours': 3600}
                    seconds = float(value) * multiplier.get(unit, 1)
                    self.performance_metrics['processing_times'].append(seconds)
            except:
                pass
        
        # Extract memory usage if available
        if 'memory' in line.lower() or 'ram' in line.lower():
            try:
                import re
                mem_pattern = r'(\d+\.?\d*)\s*(mb|gb|kb)'
                match = re.search(mem_pattern, line.lower())
                if match:
                    value, unit = match.groups()
                    # Convert to MB
                    multiplier = {'kb': 0.001, 'mb': 1, 'gb': 1024}
                    mb = float(value) * multiplier.get(unit, 1)
                    self.performance_metrics['memory_usage'].append(mb)
            except:
                pass
    
    def _check_patterns(self, line: str):
        """Check for specific error patterns."""
        error_patterns = [
            ('cuda out of memory', 'GPU memory exhausted'),
            ('connection refused', 'Connection issue detected'),
            ('timeout', 'Operation timeout detected'),
            ('permission denied', 'Permission issue detected'),
            ('no such file', 'File not found error'),
            ('failed to load', 'Resource loading failure'),
        ]
        
        line_lower = line.lower()
        for pattern, description in error_patterns:
            if pattern in line_lower:
                self._trigger_alert(f"{description}: {line}")
    
    def _check_error_rate(self):
        """Check if error rate exceeds threshold."""
        now = datetime.now()
        # Count errors in the last minute
        recent_errors = [t for t in self.recent_errors 
                        if (now - t).total_seconds() < 60]
        
        if len(recent_errors) >= self.error_threshold:
            self._trigger_alert(f"High error rate detected: {len(recent_errors)} errors in the last minute")
    
    def _check_warning_rate(self):
        """Check if warning rate exceeds threshold."""
        now = datetime.now()
        # Count warnings in the last minute
        recent_warnings = [t for t in self.recent_warnings 
                          if (now - t).total_seconds() < 60]
        
        if len(recent_warnings) >= self.warning_threshold:
            self._trigger_alert(f"High warning rate detected: {len(recent_warnings)} warnings in the last minute")
    
    def _trigger_alert(self, message: str):
        """Trigger an alert with the given message."""
        alert_message = f"ALERT [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {message}"
        
        # Log the alert
        self.logger.warning(alert_message)
        
        # Call custom alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_message)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        stats = self.stats.copy()
        
        # Calculate runtime
        if stats['start_time']:
            runtime = (datetime.now() - stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            stats['runtime_formatted'] = format_time(runtime)
        
        # Calculate rates
        if stats.get('runtime_seconds', 0) > 0:
            stats['lines_per_second'] = stats['total_lines'] / stats['runtime_seconds']
            stats['errors_per_minute'] = (stats['error_count'] / stats['runtime_seconds']) * 60
            stats['warnings_per_minute'] = (stats['warning_count'] / stats['runtime_seconds']) * 60
        
        # Add performance metrics summary
        if self.performance_metrics['processing_times']:
            times = self.performance_metrics['processing_times']
            stats['avg_processing_time'] = sum(times) / len(times)
            stats['max_processing_time'] = max(times)
            stats['min_processing_time'] = min(times)
        
        if self.performance_metrics['memory_usage']:
            memory = self.performance_metrics['memory_usage']
            stats['avg_memory_usage_mb'] = sum(memory) / len(memory)
            stats['max_memory_usage_mb'] = max(memory)
        
        return stats
    
    def print_stats(self):
        """Print current monitoring statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("LOG MONITORING STATISTICS")
        print("="*60)
        print(f"Runtime: {stats.get('runtime_formatted', 'N/A')}")
        print(f"Total log lines processed: {stats['total_lines']}")
        print(f"Lines per second: {stats.get('lines_per_second', 0):.2f}")
        print()
        print("Log Level Counts:")
        print(f"  INFO: {stats['info_count']}")
        print(f"  WARNING: {stats['warning_count']}")
        print(f"  ERROR: {stats['error_count']}")
        print(f"  CRITICAL: {stats['critical_count']}")
        print()
        print("Error/Warning Rates:")
        print(f"  Errors per minute: {stats.get('errors_per_minute', 0):.2f}")
        print(f"  Warnings per minute: {stats.get('warnings_per_minute', 0):.2f}")
        
        if 'avg_processing_time' in stats:
            print()
            print("Performance Metrics:")
            print(f"  Average processing time: {format_time(stats['avg_processing_time'])}")
            print(f"  Max processing time: {format_time(stats['max_processing_time'])}")
            
        if 'avg_memory_usage_mb' in stats:
            print(f"  Average memory usage: {stats['avg_memory_usage_mb']:.1f} MB")
            print(f"  Max memory usage: {stats['max_memory_usage_mb']:.1f} MB")
        
        print("="*60 + "\n")
    
    def get_recent_errors(self, count: int = 10) -> List[str]:
        """Get recent error log lines."""
        errors = []
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-1000:]):  # Check last 1000 lines
                        if ' - ERROR - ' in line or ' - CRITICAL - ' in line:
                            errors.append(line.strip())
                            if len(errors) >= count:
                                break
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
        
        return errors
    
    def export_stats(self, filepath: str):
        """Export monitoring statistics to JSON file."""
        stats = self.get_stats()
        
        # Convert datetime objects to strings for JSON serialization
        for key, value in stats.items():
            if isinstance(value, datetime):
                stats[key] = value.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Statistics exported to {filepath}")


def create_log_monitor(log_file: str = "logs/fusion_cuisine.log") -> LogMonitor:
    """Create and configure a log monitor instance."""
    monitor = LogMonitor(log_file)
    
    # Add default alert handler that prints to console
    def console_alert_handler(message: str):
        print(f"\n🚨 {message}\n")
    
    monitor.add_alert_handler(console_alert_handler)
    
    return monitor


def monitor_logs_command():
    """
    Command-line interface for log monitoring.
    Usage: python -c "from app.utils import monitor_logs_command; monitor_logs_command()"
    """
    print("Starting log monitoring...")
    monitor = create_log_monitor()
    
    try:
        monitor.start_monitoring()
        print("Log monitoring active. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(30)  # Print stats every 30 seconds
            monitor.print_stats()
            
    except KeyboardInterrupt:
        print("\nStopping log monitoring...")
        monitor.stop_monitoring()
        print("Final statistics:")
        monitor.print_stats()
        
        # Export final stats
        monitor.export_stats("logs/monitoring_stats.json")
        print("Monitoring statistics saved to logs/monitoring_stats.json")