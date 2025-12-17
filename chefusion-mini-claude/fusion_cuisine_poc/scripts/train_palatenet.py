#!/usr/bin/env python3
"""PalateNet training script - GraphSAGE + MLP for rating prediction."""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import SAGEConv, global_mean_pool
from tqdm import tqdm

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.utils import (
    compute_spearman_correlation,
    get_device,
    get_model_size,
    load_json,
    load_pickle,
    set_random_seeds,
    setup_logging,
)


class GraphSAGE(nn.Module):
    """GraphSAGE model for ingredient graph processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Last layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(config.get("models.palatenet.dropout", 0.2))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Not the last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class PalateNet(nn.Module):
    """Complete PalateNet model for recipe rating prediction."""
    
    def __init__(self, graph_input_dim: int, recipe_feature_dim: int, 
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        # Graph encoder for ingredients
        self.graph_encoder = GraphSAGE(
            input_dim=graph_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Recipe feature encoder
        self.recipe_encoder = nn.Sequential(
            nn.Linear(recipe_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Rating predictor
        self.rating_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output 0-1, will be scaled to 1-5
        )
        
        # Taste predictor (5 taste dimensions: sweet, sour, salty, bitter, umami)
        self.taste_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 5),
            nn.Sigmoid()  # Each taste is 0-1 intensity
        )
    
    def forward(self, graph_data: torch.Tensor, edge_index: torch.Tensor, 
                recipe_features: torch.Tensor, ingredient_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process ingredient graph
        graph_embeddings = self.graph_encoder(graph_data, edge_index)
        
        # Pool ingredient embeddings for each recipe
        recipe_graph_features = self._pool_ingredient_features(graph_embeddings, ingredient_mask)
        
        # Process recipe features
        recipe_encoded = self.recipe_encoder(recipe_features)
        
        # Fuse features
        fused_features = self.fusion(torch.cat([recipe_graph_features, recipe_encoded], dim=1))
        
        # Predict rating and taste
        rating = self.rating_predictor(fused_features).squeeze(-1)
        taste = self.taste_predictor(fused_features)
        
        # Scale rating from 0-1 to 1-5
        rating = rating * 4 + 1
        
        return rating, taste
    
    def _pool_ingredient_features(self, graph_embeddings: torch.Tensor, 
                                 ingredient_mask: torch.Tensor) -> torch.Tensor:
        """Pool ingredient embeddings for each recipe based on ingredient mask."""
        batch_size = ingredient_mask.size(0)
        hidden_dim = graph_embeddings.size(1)
        
        pooled_features = torch.zeros(batch_size, hidden_dim, device=graph_embeddings.device)
        
        for i in range(batch_size):
            mask = ingredient_mask[i]
            if mask.sum() > 0:
                # Average pool the ingredient embeddings for this recipe
                pooled_features[i] = graph_embeddings[mask.bool()].mean(dim=0)
        
        return pooled_features


class RecipeRatingDataset(Dataset):
    """Dataset for recipe rating prediction."""
    
    def __init__(self, recipes_df: pd.DataFrame, ratings_df: pd.DataFrame,
                 text_embeddings: np.ndarray, recipe_ingredient_matrix: np.ndarray,
                 ingredient_to_idx: Dict[str, int]):
        self.recipes_df = recipes_df
        self.ratings_df = ratings_df
        self.text_embeddings = text_embeddings
        self.recipe_ingredient_matrix = recipe_ingredient_matrix
        self.ingredient_to_idx = ingredient_to_idx
        
        # Create recipe to ratings mapping
        self.recipe_ratings = {}
        for _, rating in ratings_df.iterrows():
            recipe_id = rating["recipe_id"]
            if recipe_id not in self.recipe_ratings:
                self.recipe_ratings[recipe_id] = []
            self.recipe_ratings[recipe_id].append(rating["rating"])
        
        # Calculate average ratings for each recipe
        for recipe_id in self.recipe_ratings:
            self.recipe_ratings[recipe_id] = np.mean(self.recipe_ratings[recipe_id])
        
        # Filter recipes that have ratings
        self.valid_indices = []
        for idx, recipe in recipes_df.iterrows():
            if recipe["id"] in self.recipe_ratings:
                self.valid_indices.append(idx)
        
        logging.info(f"Dataset has {len(self.valid_indices)} recipes with ratings")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.valid_indices[idx]
        recipe = self.recipes_df.iloc[real_idx]
        
        # Get text features
        text_features = torch.tensor(self.text_embeddings[real_idx], dtype=torch.float32)
        
        # Get ingredient mask
        ingredient_mask = torch.tensor(self.recipe_ingredient_matrix[real_idx], dtype=torch.float32)
        
        # Get rating
        rating = self.recipe_ratings[recipe["id"]]
        
        # Generate synthetic taste profile based on cuisine and ingredients
        taste_profile = self._generate_taste_profile(recipe)
        
        return {
            "text_features": text_features,
            "ingredient_mask": ingredient_mask,
            "rating": torch.tensor(rating, dtype=torch.float32),
            "taste": torch.tensor(taste_profile, dtype=torch.float32),
            "recipe_id": recipe["id"],
            "cuisine": recipe["cuisine"]
        }
    
    def _generate_taste_profile(self, recipe: pd.Series) -> List[float]:
        """Generate synthetic taste profile based on cuisine and ingredients."""
        # Base taste profiles for cuisines
        cuisine_profiles = {
            "japanese": [0.3, 0.4, 0.7, 0.2, 0.9],  # sweet, sour, salty, bitter, umami
            "italian": [0.4, 0.6, 0.6, 0.3, 0.7]
        }
        
        base_profile = cuisine_profiles.get(recipe["cuisine"], [0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Add some noise for variation
        noise = np.random.normal(0, 0.1, 5)
        profile = np.clip(np.array(base_profile) + noise, 0, 1)
        
        return profile.tolist()


def create_recipe_features(recipes_df: pd.DataFrame, text_embeddings: np.ndarray) -> torch.Tensor:
    """Create comprehensive recipe features."""
    features = []
    
    for idx, recipe in recipes_df.iterrows():
        recipe_features = []
        
        # Add text embeddings
        recipe_features.extend(text_embeddings[idx])
        
        # Add cuisine encoding
        for cuisine in config.target_cuisines:
            recipe_features.append(1.0 if recipe["cuisine"] == cuisine else 0.0)
        
        # Add numerical features
        recipe_features.append(recipe.get("num_ingredients", 5) / 10.0)  # Normalized
        recipe_features.append(recipe.get("prep_time", 30) / 120.0)  # Normalized
        recipe_features.append(recipe.get("cook_time", 30) / 180.0)  # Normalized
        
        features.append(recipe_features)
    
    return torch.tensor(features, dtype=torch.float32)


def train_epoch(model: PalateNet, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                graph_data: torch.Tensor, edge_index: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_rating_loss = 0.0
    total_taste_loss = 0.0
    
    rating_criterion = nn.MSELoss()
    taste_criterion = nn.MSELoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        text_features = batch["text_features"].to(device)
        ingredient_mask = batch["ingredient_mask"].to(device)
        target_rating = batch["rating"].to(device)
        target_taste = batch["taste"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_rating, pred_taste = model(graph_data, edge_index, text_features, ingredient_mask)
        
        # Compute losses
        rating_loss = rating_criterion(pred_rating, target_rating)
        taste_loss = taste_criterion(pred_taste, target_taste)
        
        # Combined loss
        total_loss = rating_loss + 0.5 * taste_loss  # Weight taste loss lower
        
        total_loss.backward()
        optimizer.step()
        
        total_rating_loss += rating_loss.item()
        total_taste_loss += taste_loss.item()
    
    return total_rating_loss / len(dataloader), total_taste_loss / len(dataloader)


def validate_epoch(model: PalateNet, dataloader: DataLoader, 
                  graph_data: torch.Tensor, edge_index: torch.Tensor, device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    all_pred_ratings = []
    all_true_ratings = []
    total_rating_loss = 0.0
    total_taste_loss = 0.0
    
    rating_criterion = nn.MSELoss()
    taste_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            text_features = batch["text_features"].to(device)
            ingredient_mask = batch["ingredient_mask"].to(device)
            target_rating = batch["rating"].to(device)
            target_taste = batch["taste"].to(device)
            
            # Forward pass
            pred_rating, pred_taste = model(graph_data, edge_index, text_features, ingredient_mask)
            
            # Compute losses
            rating_loss = rating_criterion(pred_rating, target_rating)
            taste_loss = taste_criterion(pred_taste, target_taste)
            
            total_rating_loss += rating_loss.item()
            total_taste_loss += taste_loss.item()
            
            # Collect predictions for metrics
            all_pred_ratings.extend(pred_rating.cpu().numpy())
            all_true_ratings.extend(target_rating.cpu().numpy())
    
    # Compute metrics
    avg_rating_loss = total_rating_loss / len(dataloader)
    avg_taste_loss = total_taste_loss / len(dataloader)
    
    spearman_corr = compute_spearman_correlation(
        np.array(all_true_ratings), np.array(all_pred_ratings)
    )
    
    rmse = np.sqrt(mean_squared_error(all_true_ratings, all_pred_ratings))
    r2 = r2_score(all_true_ratings, all_pred_ratings)
    
    return {
        "rating_loss": avg_rating_loss,
        "taste_loss": avg_taste_loss,
        "spearman_correlation": spearman_corr,
        "rmse": rmse,
        "r2_score": r2
    }


def save_model(model: PalateNet, optimizer: torch.optim.Optimizer, epoch: int,
               metrics: Dict[str, float], filepath: Path) -> None:
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "hidden_dim": config.palatenet_hidden_dim,
            "num_layers": config.palatenet_layers,
        }
    }, filepath)
    logging.info(f"Model saved to {filepath}")


def main() -> None:
    """Main training function."""
    setup_logging(config.log_level)
    logging.info("Starting PalateNet training...")
    logging.info(f"Configuration: {config.config_path}")
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    # Get device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load preprocessed data
    train_df = pd.read_csv(config.data_dir / "train.csv")
    val_df = pd.read_csv(config.data_dir / "val.csv")
    ratings_df = pd.read_csv(config.data_dir / "ratings.csv")
    text_embeddings = np.load(config.data_dir / "text_embeddings.npy")
    recipe_ingredient_matrix = np.load(config.data_dir / "recipe_ingredient_matrix.npy")
    graph_data = torch.load(config.data_dir / "graph_data.pt")
    ingredient_to_idx = load_json(config.data_dir / "ingredient_to_idx.json")
    
    logging.info(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples")
    logging.info(f"Graph data: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Create datasets
    train_dataset = RecipeRatingDataset(
        train_df, ratings_df, text_embeddings[:len(train_df)], 
        recipe_ingredient_matrix[:len(train_df)], ingredient_to_idx
    )
    val_dataset = RecipeRatingDataset(
        val_df, ratings_df, 
        text_embeddings[len(train_df):len(train_df)+len(val_df)],
        recipe_ingredient_matrix[len(train_df):len(train_df)+len(val_df)], 
        ingredient_to_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.palatenet_batch_size, 
        shuffle=True, 
        num_workers=config.get("environment.num_workers", 4)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.palatenet_batch_size, 
        shuffle=False, 
        num_workers=config.get("environment.num_workers", 4)
    )
    
    # Prepare graph data
    graph_features = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    
    # Create recipe features
    all_recipes = pd.concat([train_df, val_df], ignore_index=True)
    all_text_embeddings = text_embeddings[:len(all_recipes)]
    recipe_features = create_recipe_features(all_recipes, all_text_embeddings)
    recipe_feature_dim = recipe_features.size(1)
    
    # Initialize model
    model = PalateNet(
        graph_input_dim=graph_features.size(1),
        recipe_feature_dim=recipe_feature_dim,
        hidden_dim=config.palatenet_hidden_dim,
        num_layers=config.palatenet_layers
    ).to(device)
    
    logging.info(f"Model size: {get_model_size(model)} parameters")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get("models.palatenet.learning_rate", 0.001),
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    
    # Training loop
    best_spearman = -1.0
    start_time = time.time()
    
    for epoch in range(config.palatenet_epochs):
        logging.info(f"Epoch {epoch + 1}/{config.palatenet_epochs}")
        
        # Train
        train_rating_loss, train_taste_loss = train_epoch(
            model, train_loader, optimizer, graph_features, edge_index, device
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, graph_features, edge_index, device)
        
        # Update learning rate based on Spearman correlation
        scheduler.step(val_metrics["spearman_correlation"])
        
        # Log progress
        logging.info(f"Train - Rating Loss: {train_rating_loss:.4f}, Taste Loss: {train_taste_loss:.4f}")
        logging.info(f"Val - Spearman: {val_metrics['spearman_correlation']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        # Save best model
        if val_metrics["spearman_correlation"] > best_spearman:
            best_spearman = val_metrics["spearman_correlation"]
            save_model(model, optimizer, epoch, val_metrics, config.models_dir / "palatenet_best.pt")
    
    # Save final model
    save_model(model, optimizer, config.palatenet_epochs - 1, val_metrics, config.models_dir / "palatenet_final.pt")
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Print summary
    print("\n" + "="*50)
    print("PALATENET TRAINING COMPLETE")
    print("="*50)
    print(f"🎯 Best Spearman correlation: {best_spearman:.4f}")
    print(f"📊 Target correlation: {config.target_spearman_correlation:.4f}")
    print(f"✅ Target achieved: {'Yes' if best_spearman >= config.target_spearman_correlation else 'No'}")
    print(f"📈 Final RMSE: {val_metrics['rmse']:.4f}")
    print(f"📊 Final R²: {val_metrics['r2_score']:.4f}")
    print(f"⏱️  Training time: {training_time:.1f}s")
    print(f"🧠 Model size: {get_model_size(model)}")
    print(f"💾 Model saved to: {config.models_dir}")
    print("="*50)


if __name__ == "__main__":
    main()