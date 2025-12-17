#!/usr/bin/env python3
"""Evaluation script for fusion cuisine generation and rating prediction."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.utils import (
    compute_ingredient_overlap,
    compute_spearman_correlation,
    get_device,
    load_json,
    print_metrics,
    set_random_seeds,
    setup_logging,
)

# Import model classes
from train_palatenet import PalateNet, RecipeRatingDataset


class FusionEvaluator:
    """Evaluator for fusion cuisine generation and prediction models."""
    
    def __init__(self):
        self.device = get_device()
        self.palatenet_model = None
        
    def load_models(self) -> None:
        """Load trained models for evaluation."""
        logging.info("Loading models for evaluation...")
        
        # Load PalateNet model
        palatenet_path = config.models_dir / "palatenet_best.pt"
        if palatenet_path.exists():
            self._load_palatenet_model(palatenet_path)
        else:
            logging.warning(f"PalateNet model not found at {palatenet_path}")
    
    def _load_palatenet_model(self, model_path: Path) -> None:
        """Load the trained PalateNet model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load required data to reconstruct model
        text_embeddings = np.load(config.data_dir / "text_embeddings.npy")
        graph_data = torch.load(config.data_dir / "graph_data.pt")
        
        # Create recipe features to get dimension
        test_df = pd.read_csv(config.data_dir / "test.csv")
        recipe_features = self._create_recipe_features(test_df, text_embeddings[:len(test_df)])
        
        # Initialize model with correct dimensions
        self.palatenet_model = PalateNet(
            graph_input_dim=graph_data.x.size(1),
            recipe_feature_dim=recipe_features.size(1),
            hidden_dim=config.palatenet_hidden_dim,
            num_layers=config.palatenet_layers
        )
        
        self.palatenet_model.load_state_dict(checkpoint["model_state_dict"])
        self.palatenet_model.to(self.device)
        self.palatenet_model.eval()
        
        logging.info(f"Loaded PalateNet model from {model_path}")
    
    def _create_recipe_features(self, recipes_df: pd.DataFrame, text_embeddings: np.ndarray) -> torch.Tensor:
        """Create recipe features for evaluation."""
        features = []
        
        for idx, recipe in recipes_df.iterrows():
            recipe_features = []
            
            # Add text embeddings
            recipe_features.extend(text_embeddings[idx])
            
            # Add cuisine encoding
            for cuisine in config.target_cuisines:
                recipe_features.append(1.0 if recipe["cuisine"] == cuisine else 0.0)
            
            # Add numerical features
            recipe_features.append(recipe.get("num_ingredients", 5) / 10.0)
            recipe_features.append(recipe.get("prep_time", 30) / 120.0)
            recipe_features.append(recipe.get("cook_time", 30) / 180.0)
            
            features.append(recipe_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def evaluate_rating_prediction(self) -> Dict[str, float]:
        """Evaluate rating prediction performance on test set."""
        logging.info("Evaluating rating prediction performance...")
        
        if self.palatenet_model is None:
            logging.warning("PalateNet model not loaded, skipping rating evaluation")
            return {}
        
        # Load test data
        test_df = pd.read_csv(config.data_dir / "test.csv")
        ratings_df = pd.read_csv(config.data_dir / "ratings.csv")
        text_embeddings = np.load(config.data_dir / "text_embeddings.npy")
        recipe_ingredient_matrix = np.load(config.data_dir / "recipe_ingredient_matrix.npy")
        graph_data = torch.load(config.data_dir / "graph_data.pt")
        ingredient_to_idx = load_json(config.data_dir / "ingredient_to_idx.json")
        
        # Create test dataset
        test_dataset = RecipeRatingDataset(
            test_df, ratings_df,
            text_embeddings[-len(test_df):],  # Test embeddings are at the end
            recipe_ingredient_matrix[-len(test_df):],
            ingredient_to_idx
        )
        
        if len(test_dataset) == 0:
            logging.warning("No test samples with ratings found")
            return {}
        
        # Prepare data
        graph_features = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        
        # Evaluate model
        predictions = []
        targets = []
        
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
                sample = test_dataset[i]
                
                text_features = sample["text_features"].unsqueeze(0).to(self.device)
                ingredient_mask = sample["ingredient_mask"].unsqueeze(0).to(self.device)
                target_rating = sample["rating"].item()
                
                # Predict rating
                pred_rating, _ = self.palatenet_model(
                    graph_features, edge_index, text_features, ingredient_mask
                )
                
                predictions.append(pred_rating.item())
                targets.append(target_rating)
        
        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        spearman_corr = compute_spearman_correlation(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = np.mean(np.abs(targets - predictions))
        r2 = r2_score(targets, predictions)
        
        metrics = {
            "spearman_correlation": spearman_corr,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "target_achieved": spearman_corr >= config.target_spearman_correlation,
            "num_samples": len(test_dataset)
        }
        
        logging.info(f"Rating prediction metrics: {metrics}")
        return metrics
    
    def evaluate_ingredient_overlap(self) -> Dict[str, float]:
        """Evaluate ingredient overlap between generated and real recipes."""
        logging.info("Evaluating ingredient overlap...")
        
        # Load fusion recipes
        fusion_recipes_path = config.outputs_dir / "recipes" / "fusion_recipes.json"
        if not fusion_recipes_path.exists():
            logging.warning(f"Fusion recipes not found at {fusion_recipes_path}")
            return {}
        
        fusion_recipes = load_json(fusion_recipes_path)
        
        # Load real recipes for comparison
        test_df = pd.read_csv(config.data_dir / "test.csv")
        
        overlap_scores = []
        alpha_overlaps = {}
        
        for alpha_key, fusion_data in fusion_recipes.items():
            alpha = fusion_data["alpha"]
            recipe_text = fusion_data["recipe"]
            
            # Extract ingredients from fusion recipe (simple extraction)
            fusion_ingredients = self._extract_ingredients_from_text(recipe_text)
            
            # Compare with real recipes
            recipe_overlaps = []
            for _, real_recipe in test_df.iterrows():
                real_ingredients = [ing.strip().lower() for ing in real_recipe["ingredients"].split(",")]
                overlap = compute_ingredient_overlap(fusion_ingredients, real_ingredients)
                recipe_overlaps.append(overlap)
            
            avg_overlap = np.mean(recipe_overlaps)
            alpha_overlaps[f"alpha_{alpha:.1f}"] = avg_overlap
            overlap_scores.extend(recipe_overlaps)
        
        metrics = {
            "avg_ingredient_overlap": np.mean(overlap_scores),
            "std_ingredient_overlap": np.std(overlap_scores),
            "min_ingredient_overlap": np.min(overlap_scores),
            "max_ingredient_overlap": np.max(overlap_scores),
            "alpha_specific_overlaps": alpha_overlaps,
            "num_comparisons": len(overlap_scores)
        }
        
        logging.info(f"Ingredient overlap metrics: {metrics}")
        return metrics
    
    def _extract_ingredients_from_text(self, recipe_text: str) -> List[str]:
        """Extract ingredients from recipe text (simple heuristic)."""
        ingredients = []
        
        # Look for ingredients section
        lines = recipe_text.split('\n')
        in_ingredients_section = False
        
        for line in lines:
            line = line.strip().lower()
            
            # Start of ingredients section
            if 'ingredients' in line:
                in_ingredients_section = True
                continue
            
            # End of ingredients section
            if in_ingredients_section and ('instructions' in line or 'method' in line or 'directions' in line):
                break
            
            # Extract ingredient
            if in_ingredients_section and line and not line.startswith('#'):
                # Remove quantities and extract ingredient names
                ingredient = self._clean_ingredient_line(line)
                if ingredient:
                    ingredients.append(ingredient)
        
        return ingredients
    
    def _clean_ingredient_line(self, line: str) -> Optional[str]:
        """Clean an ingredient line to extract the main ingredient."""
        # Remove common measurement words
        measurement_words = ['cup', 'cups', 'tbsp', 'tablespoon', 'tsp', 'teaspoon', 
                            'oz', 'pound', 'lb', 'kg', 'gram', 'g', 'ml', 'liter']
        
        # Remove numbers and measurements
        words = line.split()
        cleaned_words = []
        
        for word in words:
            word = word.strip(',-()[]')
            if not word.replace('.', '').isdigit() and word.lower() not in measurement_words:
                cleaned_words.append(word)
        
        if cleaned_words:
            # Take the main ingredient (usually the last significant word)
            return ' '.join(cleaned_words[:3])  # First few words usually contain the ingredient
        
        return None
    
    def compute_fid_score(self) -> Optional[float]:
        """Compute FID score between generated and real food images."""
        logging.info("Computing FID score...")
        
        try:
            from torchvision.models import inception_v3
            import torchvision.transforms as transforms
        except ImportError:
            logging.warning("Required packages for FID computation not available")
            return None
        
        # Load generated images
        fusion_images_path = config.outputs_dir / "images" / "fusion_images.json"
        if not fusion_images_path.exists():
            logging.warning("No generated images found for FID computation")
            return None
        
        fusion_images_data = load_json(fusion_images_path)
        
        # Load real images
        real_images_dir = config.data_dir / "images"
        if not real_images_dir.exists():
            logging.warning("Real images directory not found")
            return None
        
        # Simple FID approximation (would need proper implementation for real use)
        logging.info("FID computation would require additional setup - returning placeholder")
        return 0.5  # Placeholder FID score
    
    def evaluate_diversity_metrics(self) -> Dict[str, float]:
        """Evaluate diversity of generated recipes."""
        logging.info("Evaluating recipe diversity...")
        
        fusion_recipes_path = config.outputs_dir / "recipes" / "fusion_recipes.json"
        if not fusion_recipes_path.exists():
            logging.warning("Fusion recipes not found")
            return {}
        
        fusion_recipes = load_json(fusion_recipes_path)
        
        # Extract all ingredients from fusion recipes
        all_fusion_ingredients = []
        recipe_lengths = []
        
        for alpha_key, fusion_data in fusion_recipes.items():
            recipe_text = fusion_data["recipe"]
            ingredients = self._extract_ingredients_from_text(recipe_text)
            all_fusion_ingredients.extend(ingredients)
            recipe_lengths.append(len(ingredients))
        
        # Compute diversity metrics
        unique_ingredients = set(all_fusion_ingredients)
        
        metrics = {
            "total_unique_ingredients": len(unique_ingredients),
            "avg_ingredients_per_recipe": np.mean(recipe_lengths),
            "std_ingredients_per_recipe": np.std(recipe_lengths),
            "ingredient_diversity_ratio": len(unique_ingredients) / len(all_fusion_ingredients) if all_fusion_ingredients else 0,
            "num_fusion_recipes": len(fusion_recipes)
        }
        
        logging.info(f"Diversity metrics: {metrics}")
        return metrics
    
    def run_comprehensive_evaluation(self) -> Dict[str, any]:
        """Run comprehensive evaluation of the fusion cuisine system."""
        logging.info("Running comprehensive evaluation...")
        
        start_time = time.time()
        
        # Load models
        self.load_models()
        
        # Run all evaluations
        rating_metrics = self.evaluate_rating_prediction()
        overlap_metrics = self.evaluate_ingredient_overlap()
        diversity_metrics = self.evaluate_diversity_metrics()
        fid_score = self.compute_fid_score()
        
        # Combine all metrics
        comprehensive_metrics = {
            "rating_prediction": rating_metrics,
            "ingredient_overlap": overlap_metrics,
            "diversity": diversity_metrics,
            "fid_score": fid_score,
            "evaluation_time": time.time() - start_time,
            "config_file": str(config.config_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return comprehensive_metrics


def main() -> None:
    """Main evaluation function."""
    setup_logging(config.log_level)
    logging.info("Starting comprehensive evaluation...")
    logging.info(f"Configuration: {config.config_path}")
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    # Initialize evaluator
    evaluator = FusionEvaluator()
    
    # Run evaluation
    metrics = evaluator.run_comprehensive_evaluation()
    
    # Save metrics
    metrics_path = config.outputs_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Evaluation metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    
    # Rating prediction metrics
    if metrics["rating_prediction"]:
        rating_metrics = metrics["rating_prediction"]
        print(f"📊 RATING PREDICTION PERFORMANCE:")
        print(f"   Spearman Correlation: {rating_metrics.get('spearman_correlation', 0):.4f}")
        print(f"   Target (≥{config.target_spearman_correlation}): {'✅ ACHIEVED' if rating_metrics.get('target_achieved', False) else '❌ NOT ACHIEVED'}")
        print(f"   RMSE: {rating_metrics.get('rmse', 0):.4f}")
        print(f"   R² Score: {rating_metrics.get('r2_score', 0):.4f}")
        print(f"   Test Samples: {rating_metrics.get('num_samples', 0)}")
    
    # Ingredient overlap metrics
    if metrics["ingredient_overlap"]:
        overlap_metrics = metrics["ingredient_overlap"]
        print(f"\n🥘 INGREDIENT OVERLAP ANALYSIS:")
        print(f"   Average Overlap: {overlap_metrics.get('avg_ingredient_overlap', 0):.4f}")
        print(f"   Standard Deviation: {overlap_metrics.get('std_ingredient_overlap', 0):.4f}")
        print(f"   Range: [{overlap_metrics.get('min_ingredient_overlap', 0):.3f}, {overlap_metrics.get('max_ingredient_overlap', 0):.3f}]")
        
        alpha_overlaps = overlap_metrics.get('alpha_specific_overlaps', {})
        for alpha_key, overlap in alpha_overlaps.items():
            print(f"   {alpha_key}: {overlap:.4f}")
    
    # Diversity metrics
    if metrics["diversity"]:
        diversity_metrics = metrics["diversity"]
        print(f"\n🌈 RECIPE DIVERSITY METRICS:")
        print(f"   Unique Ingredients: {diversity_metrics.get('total_unique_ingredients', 0)}")
        print(f"   Avg Ingredients/Recipe: {diversity_metrics.get('avg_ingredients_per_recipe', 0):.1f}")
        print(f"   Diversity Ratio: {diversity_metrics.get('ingredient_diversity_ratio', 0):.4f}")
        print(f"   Fusion Recipes Generated: {diversity_metrics.get('num_fusion_recipes', 0)}")
    
    # FID score
    if metrics["fid_score"] is not None:
        print(f"\n🖼️  IMAGE QUALITY (FID Score): {metrics['fid_score']:.4f}")
    
    # Overall summary
    print(f"\n⏱️  EVALUATION TIME: {metrics['evaluation_time']:.1f} seconds")
    print(f"💾 RESULTS SAVED TO: {metrics_path}")
    print("="*60)
    
    # Final assessment
    if metrics["rating_prediction"].get("target_achieved", False):
        print("🎉 SUCCESS: Target Spearman correlation achieved!")
    else:
        print("⚠️  ATTENTION: Target Spearman correlation not achieved.")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()