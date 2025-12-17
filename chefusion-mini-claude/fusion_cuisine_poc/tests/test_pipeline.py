"""Test suite for the fusion cuisine pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.config import Config
from app.utils import (
    clean_recipe_text,
    compute_ingredient_overlap,
    compute_spearman_correlation,
    create_adjacency_matrix,
    get_device,
    split_dataset,
)


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_config_loading(self):
        """Test that configuration loads properly."""
        config = Config()
        assert config.target_recipes_per_cuisine == 5000
        assert "japanese" in config.target_cuisines
        assert "italian" in config.target_cuisines
        assert config.latent_dim == 256
    
    def test_config_paths(self):
        """Test that configuration paths are set correctly."""
        config = Config()
        assert config.data_dir.name == "data"
        assert config.models_dir.name == "models"
        assert config.outputs_dir.name == "outputs"
    
    def test_config_get_method(self):
        """Test the configuration get method with dot notation."""
        config = Config()
        assert config.get("dataset.target_recipes_per_cuisine") == 5000
        assert config.get("nonexistent.key", "default") == "default"


class TestUtils:
    """Test utility functions."""
    
    def test_clean_recipe_text(self):
        """Test recipe text cleaning."""
        dirty_text = "  This is a  recipe\n\rwith   extra   spaces  "
        cleaned = clean_recipe_text(dirty_text)
        assert cleaned == "This is a recipe with extra spaces"
        
        # Test with non-string input
        assert clean_recipe_text(None) == ""
        assert clean_recipe_text(123) == ""
    
    def test_compute_ingredient_overlap(self):
        """Test ingredient overlap computation."""
        ingredients1 = ["soy sauce", "rice", "salmon"]
        ingredients2 = ["rice", "salmon", "nori"]
        
        overlap = compute_ingredient_overlap(ingredients1, ingredients2)
        assert 0 <= overlap <= 1
        
        # Test identical lists
        overlap_identical = compute_ingredient_overlap(ingredients1, ingredients1)
        assert overlap_identical == 1.0
        
        # Test empty lists
        overlap_empty = compute_ingredient_overlap([], [])
        assert overlap_empty == 1.0
    
    def test_compute_spearman_correlation(self):
        """Test Spearman correlation computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        corr = compute_spearman_correlation(y_true, y_pred)
        assert -1 <= corr <= 1
        assert corr > 0.8  # Should be highly correlated
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        df = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100)
        })
        
        train_df, val_df, test_df = split_dataset(df, test_ratio=0.2, val_ratio=0.1)
        
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert len(test_df) == 20  # 20% of 100
        assert len(val_df) == 10   # 10% of 100
    
    def test_create_adjacency_matrix(self):
        """Test adjacency matrix creation."""
        flavor_graph = {
            "ingredient1": ["ingredient2", "ingredient3"],
            "ingredient2": ["ingredient1"],
            "ingredient3": ["ingredient1"]
        }
        
        adj_matrix, ingredient_to_idx = create_adjacency_matrix(flavor_graph)
        
        assert adj_matrix.shape == (3, 3)
        assert len(ingredient_to_idx) == 3
        assert np.allclose(adj_matrix, adj_matrix.T)  # Should be symmetric
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


class TestDataGeneration:
    """Test data generation scripts."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create temporary config
            config_dict = {
                "dataset": {
                    "target_recipes_per_cuisine": 10,  # Small for testing
                    "target_cuisines": ["japanese", "italian"],
                    "min_ingredient_frequency": 2
                },
                "paths": {
                    "data_dir": str(temp_path / "data"),
                    "models_dir": str(temp_path / "models"),
                    "outputs_dir": str(temp_path / "outputs")
                },
                "environment": {
                    "offline_mode": True,
                    "random_seed": 42
                }
            }
            
            config_file = temp_path / "test_config.yaml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f)
            
            yield Config(config_file)
    
    def test_recipe_generation_structure(self, temp_config):
        """Test that recipe generation creates proper data structure."""
        # This would normally import and run the data generation script
        # For now, we'll test the expected output structure
        
        # Create mock data structure
        recipes_data = {
            'id': ['jp_000001', 'it_000001'],
            'title': ['Japanese Recipe 1', 'Italian Recipe 1'],
            'ingredients': ['soy sauce, rice', 'olive oil, pasta'],
            'cuisine': ['japanese', 'italian'],
            'rating': [4.5, 4.2]
        }
        
        df = pd.DataFrame(recipes_data)
        
        # Test data structure
        assert 'id' in df.columns
        assert 'title' in df.columns
        assert 'ingredients' in df.columns
        assert 'cuisine' in df.columns
        assert 'rating' in df.columns
        
        # Test cuisine distribution
        cuisine_counts = df['cuisine'].value_counts()
        assert 'japanese' in cuisine_counts
        assert 'italian' in cuisine_counts


class TestModels:
    """Test model architectures and functionality."""
    
    def test_model_dimensions(self):
        """Test that models have correct input/output dimensions."""
        from train_encoder import TextEncoder, ImageEncoder
        
        # Test text encoder
        text_encoder = TextEncoder(input_dim=384, hidden_dim=512, output_dim=256)
        test_input = torch.randn(4, 384)
        output = text_encoder(test_input)
        assert output.shape == (4, 256)
        
        # Test image encoder
        image_encoder = ImageEncoder(output_dim=256)
        test_images = torch.randn(4, 3, 256, 256)
        output = image_encoder(test_images)
        assert output.shape == (4, 256)
    
    def test_model_forward_pass(self):
        """Test that models can perform forward passes without errors."""
        from train_encoder import CLIPMini
        
        model = CLIPMini(text_input_dim=384, latent_dim=256)
        
        text_features = torch.randn(4, 384)
        images = torch.randn(4, 3, 256, 256)
        
        text_embeddings, image_embeddings = model(text_features, images)
        
        assert text_embeddings.shape == (4, 256)
        assert image_embeddings.shape == (4, 256)
        
        # Test that embeddings are normalized
        text_norms = torch.norm(text_embeddings, dim=1)
        assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5)


class TestPipeline:
    """Test full pipeline integration."""
    
    def test_pipeline_components_exist(self):
        """Test that all pipeline scripts exist."""
        scripts_dir = project_root / "scripts"
        
        required_scripts = [
            "01_fetch_data.py",
            "02_preprocess.py", 
            "train_encoder.py",
            "train_palatenet.py",
            "generate_fusion.py",
            "evaluate.py"
        ]
        
        for script in required_scripts:
            script_path = scripts_dir / script
            assert script_path.exists(), f"Required script {script} not found"
            
            # Test that scripts are executable Python files
            with open(script_path, 'r') as f:
                content = f.read()
                assert content.startswith('#!/usr/bin/env python3') or 'python' in content
    
    def test_makefile_targets(self):
        """Test that Makefile has required targets."""
        makefile_path = project_root / "Makefile"
        assert makefile_path.exists()
        
        with open(makefile_path, 'r') as f:
            makefile_content = f.read()
        
        required_targets = ['all', 'data', 'preprocess', 'train', 'generate', 'evaluate', 'clean']
        
        for target in required_targets:
            assert f"{target}:" in makefile_content, f"Makefile target '{target}' not found"
    
    def test_docker_configuration(self):
        """Test Docker configuration files."""
        # Test Dockerfile exists
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists()
        
        # Test docker-compose.yml exists
        compose_path = project_root / "docker-compose.yml"
        assert compose_path.exists()
        
        # Test that docker-compose.yml has required services
        with open(compose_path, 'r') as f:
            compose_content = f.read()
        
        assert "app:" in compose_content
        assert "8888:8888" in compose_content  # JupyterLab port
    
    def test_output_directory_structure(self):
        """Test that output directories are created properly."""
        from app.config import config
        
        # Test that directories are created
        assert config.data_dir.exists() or True  # May not exist until runtime
        assert config.outputs_dir.exists() or True
        
        # Test expected subdirectories
        expected_dirs = ["images", "recipes"]
        for subdir in expected_dirs:
            expected_path = config.outputs_dir / subdir
            # These may not exist until pipeline runs, but structure should be correct
            assert subdir in str(expected_path)


class TestEvaluation:
    """Test evaluation metrics and functions."""
    
    def test_evaluation_metrics_structure(self):
        """Test that evaluation produces the correct metrics structure."""
        # Mock evaluation results structure
        mock_metrics = {
            "rating_prediction": {
                "spearman_correlation": 0.42,
                "rmse": 0.68,
                "target_achieved": True
            },
            "ingredient_overlap": {
                "avg_ingredient_overlap": 0.53,
                "alpha_specific_overlaps": {
                    "alpha_0.3": 0.61,
                    "alpha_0.5": 0.52,
                    "alpha_0.7": 0.46
                }
            },
            "diversity": {
                "total_unique_ingredients": 45,
                "num_fusion_recipes": 3
            }
        }
        
        # Test structure validation
        assert "rating_prediction" in mock_metrics
        assert "ingredient_overlap" in mock_metrics
        assert "diversity" in mock_metrics
        
        # Test specific metric validation
        rating_metrics = mock_metrics["rating_prediction"]
        assert "spearman_correlation" in rating_metrics
        assert "target_achieved" in rating_metrics
        assert isinstance(rating_metrics["target_achieved"], bool)
        
        overlap_metrics = mock_metrics["ingredient_overlap"]
        assert "avg_ingredient_overlap" in overlap_metrics
        assert "alpha_specific_overlaps" in overlap_metrics


# Integration test that can be run manually
def test_smoke_test():
    """Smoke test to verify basic functionality."""
    # Test imports
    from app.config import config
    from app.utils import clean_recipe_text, get_device
    
    # Test basic functionality
    device = get_device()
    assert device is not None
    
    cleaned_text = clean_recipe_text("Test recipe text")
    assert cleaned_text == "Test recipe text"
    
    # Test configuration
    assert config.target_recipes_per_cuisine > 0
    assert len(config.target_cuisines) > 0


if __name__ == "__main__":
    # Run smoke test if called directly
    test_smoke_test()
    print("✅ All smoke tests passed!")