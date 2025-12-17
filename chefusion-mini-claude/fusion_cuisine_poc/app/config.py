"""Configuration settings for the fusion cuisine POC."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class Config:
    """Central configuration class that loads from config.yaml."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            # Look for config.yaml in the project root
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.outputs_dir,
            self.notebooks_dir,
            self.outputs_dir / "images",
            self.outputs_dir / "recipes",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(os.getenv("DATA_DIR", self.get("paths.data_dir", "app/data")))
    
    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return Path(os.getenv("MODEL_DIR", self.get("paths.models_dir", "app/models")))
    
    @property
    def outputs_dir(self) -> Path:
        """Get outputs directory path."""
        return Path(os.getenv("OUTPUT_DIR", self.get("paths.outputs_dir", "outputs")))
    
    @property
    def notebooks_dir(self) -> Path:
        """Get notebooks directory path."""
        return Path(self.get("paths.notebooks_dir", "app/notebooks"))
    
    @property
    def scripts_dir(self) -> Path:
        """Get scripts directory path."""
        return Path(self.get("paths.scripts_dir", "scripts"))
    
    # Dataset properties
    @property
    def target_recipes_per_cuisine(self) -> int:
        return self.get("dataset.target_recipes_per_cuisine", 5000)
    
    @property
    def target_cuisines(self) -> List[str]:
        return self.get("dataset.target_cuisines", ["japanese", "italian"])
    
    @property
    def min_ingredient_frequency(self) -> int:
        return self.get("dataset.min_ingredient_frequency", 10)
    
    @property
    def image_size(self) -> int:
        return self.get("dataset.image_size", 256)
    
    @property
    def test_split_ratio(self) -> float:
        return self.get("dataset.test_split_ratio", 0.1)
    
    # Model properties
    @property
    def latent_dim(self) -> int:
        return self.get("models.encoder.latent_dim", 256)
    
    @property
    def encoder_batch_size(self) -> int:
        return self.get("models.encoder.batch_size", 32)
    
    @property
    def encoder_epochs(self) -> int:
        return self.get("models.encoder.epochs", 10)
    
    @property
    def palatenet_hidden_dim(self) -> int:
        return self.get("models.palatenet.hidden_dim", 64)
    
    @property
    def palatenet_layers(self) -> int:
        return self.get("models.palatenet.num_layers", 2)
    
    @property
    def palatenet_batch_size(self) -> int:
        return self.get("models.palatenet.batch_size", 64)
    
    @property
    def palatenet_epochs(self) -> int:
        return self.get("models.palatenet.epochs", 20)
    
    # Fusion properties
    @property
    def fusion_alphas(self) -> List[float]:
        return self.get("fusion.alpha_values", [0.3, 0.5, 0.7])
    
    # Environment properties
    @property
    def offline_mode(self) -> bool:
        return bool(os.getenv("OFFLINE_MODE", self.get("environment.offline_mode", False)))
    
    @property
    def use_gpu(self) -> bool:
        cuda_available = os.getenv("CUDA_VISIBLE_DEVICES") is not None
        return cuda_available or self.get("environment.use_gpu", True)
    
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY", self.get("api.openai_api_key"))
    
    @property
    def target_spearman_correlation(self) -> float:
        return self.get("evaluation.target_spearman_correlation", 0.3)
    
    @property
    def random_seed(self) -> int:
        return self.get("environment.random_seed", 42)
    
    @property
    def log_level(self) -> str:
        return self.get("environment.log_level", "INFO")


# Global configuration instance
config = Config()

# Backward compatibility alias
settings = config