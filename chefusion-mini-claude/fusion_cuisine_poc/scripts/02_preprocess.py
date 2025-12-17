#!/usr/bin/env python3
"""Data preprocessing script for PyG-compatible format."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from tqdm import tqdm

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.utils import (
    create_adjacency_matrix,
    load_json,
    save_json,
    save_pickle,
    set_random_seeds,
    setup_logging,
    split_dataset,
)


class RecipePreprocessor:
    """Preprocessor for recipe data."""
    
    def __init__(self):
        self.sentence_transformer = None
        self.ingredient_encoder = LabelEncoder()
        self.cuisine_encoder = LabelEncoder()
        self.ingredient_to_idx = {}
        self.idx_to_ingredient = {}
        
    def load_sentence_transformer(self) -> None:
        """Load sentence transformer model."""
        logging.info("Loading sentence transformer model...")
        model_name = config.get("preprocessing.tokenizer_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.sentence_transformer = SentenceTransformer(model_name)
        logging.info(f"Loaded sentence transformer: {model_name}")
    
    def process_text_features(self, recipes_df: pd.DataFrame) -> np.ndarray:
        """Process text features using sentence transformer."""
        logging.info("Processing text features...")
        
        if self.sentence_transformer is None:
            self.load_sentence_transformer()
        
        # Combine title and instructions for richer text representation
        combined_text = (
            recipes_df["title"] + " " + recipes_df["instructions"]
        ).tolist()
        
        # Generate embeddings
        embeddings = self.sentence_transformer.encode(
            combined_text,
            show_progress_bar=True,
            batch_size=32
        )
        
        logging.info(f"Generated {len(embeddings)} text embeddings with dimension {embeddings.shape[1]}")
        return embeddings
    
    def process_ingredient_features(self, recipes_df: pd.DataFrame, flavor_graph: Dict[str, List[str]]) -> Tuple[np.ndarray, Dict[str, int]]:
        """Process ingredient features and create ingredient embeddings."""
        logging.info("Processing ingredient features...")
        
        # Get all unique ingredients from flavor graph
        all_ingredients = list(flavor_graph.keys())
        
        # Create ingredient to index mapping
        self.ingredient_to_idx = {ingredient: i for i, ingredient in enumerate(all_ingredients)}
        self.idx_to_ingredient = {i: ingredient for ingredient, i in self.ingredient_to_idx.items()}
        
        # Create ingredient embeddings using sentence transformer
        if self.sentence_transformer is None:
            self.load_sentence_transformer()
        
        ingredient_embeddings = self.sentence_transformer.encode(
            all_ingredients,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create recipe-ingredient matrix
        n_recipes = len(recipes_df)
        n_ingredients = len(all_ingredients)
        recipe_ingredient_matrix = np.zeros((n_recipes, n_ingredients), dtype=np.float32)
        
        for i, recipe in tqdm(recipes_df.iterrows(), total=n_recipes, desc="Processing recipe ingredients"):
            recipe_ingredients = [ing.strip().lower() for ing in recipe["ingredients"].split(",")]
            for ingredient in recipe_ingredients:
                if ingredient in self.ingredient_to_idx:
                    j = self.ingredient_to_idx[ingredient]
                    recipe_ingredient_matrix[i, j] = 1.0
        
        logging.info(f"Created recipe-ingredient matrix: {recipe_ingredient_matrix.shape}")
        logging.info(f"Created ingredient embeddings: {ingredient_embeddings.shape}")
        
        return ingredient_embeddings, recipe_ingredient_matrix
    
    def create_graph_data(self, flavor_graph: Dict[str, List[str]], ingredient_embeddings: np.ndarray) -> Data:
        """Create PyTorch Geometric Data object from flavor graph."""
        logging.info("Creating PyG graph data...")
        
        # Create adjacency matrix
        adj_matrix, ingredient_to_idx = create_adjacency_matrix(flavor_graph)
        
        # Convert to edge index format for PyG
        edge_indices = np.nonzero(adj_matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # Create node features (ingredient embeddings)
        node_features = torch.tensor(ingredient_embeddings, dtype=torch.float32)
        
        # Create edge weights (all 1.0 for now, could be enhanced with frequency counts)
        edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32)
        
        # Create PyG Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            num_nodes=len(ingredient_to_idx)
        )
        
        logging.info(f"Created graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
        
        return graph_data
    
    def encode_categorical_features(self, recipes_df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logging.info("Encoding categorical features...")
        
        # Create a copy to avoid modifying original
        processed_df = recipes_df.copy()
        
        # Encode cuisine labels
        processed_df["cuisine_encoded"] = self.cuisine_encoder.fit_transform(processed_df["cuisine"])
        
        # Create binary cuisine indicators
        for cuisine in config.target_cuisines:
            processed_df[f"is_{cuisine}"] = (processed_df["cuisine"] == cuisine).astype(int)
        
        logging.info(f"Encoded cuisines: {list(self.cuisine_encoder.classes_)}")
        
        return processed_df
    
    def create_dataset_splits(self, recipes_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train, validation, and test splits."""
        logging.info("Creating dataset splits...")
        
        # Stratified split by cuisine to ensure balanced representation
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for cuisine in config.target_cuisines:
            cuisine_df = recipes_df[recipes_df["cuisine"] == cuisine]
            
            train_df, val_df, test_df = split_dataset(
                cuisine_df,
                test_ratio=config.test_split_ratio,
                val_ratio=config.get("dataset.validation_split_ratio", 0.1),
                random_state=config.random_seed
            )
            
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)
        
        # Combine all cuisine splits
        train_combined = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=config.random_seed)
        val_combined = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=config.random_seed)
        test_combined = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=config.random_seed)
        
        logging.info(f"Dataset splits - Train: {len(train_combined)}, Val: {len(val_combined)}, Test: {len(test_combined)}")
        
        return train_combined, val_combined, test_combined
    
    def save_preprocessed_data(self, 
                              train_df: pd.DataFrame, 
                              val_df: pd.DataFrame, 
                              test_df: pd.DataFrame,
                              text_embeddings: np.ndarray,
                              ingredient_embeddings: np.ndarray,
                              recipe_ingredient_matrix: np.ndarray,
                              graph_data: Data) -> None:
        """Save all preprocessed data."""
        logging.info("Saving preprocessed data...")
        
        # Save dataframes
        train_df.to_csv(config.data_dir / "train.csv", index=False)
        val_df.to_csv(config.data_dir / "val.csv", index=False)
        test_df.to_csv(config.data_dir / "test.csv", index=False)
        
        # Save embeddings and matrices
        np.save(config.data_dir / "text_embeddings.npy", text_embeddings)
        np.save(config.data_dir / "ingredient_embeddings.npy", ingredient_embeddings)
        np.save(config.data_dir / "recipe_ingredient_matrix.npy", recipe_ingredient_matrix)
        
        # Save graph data
        torch.save(graph_data, config.data_dir / "graph_data.pt")
        
        # Save encoders and mappings
        save_pickle(self.ingredient_encoder, config.data_dir / "ingredient_encoder.pkl")
        save_pickle(self.cuisine_encoder, config.data_dir / "cuisine_encoder.pkl")
        save_json(self.ingredient_to_idx, config.data_dir / "ingredient_to_idx.json")
        save_json(self.idx_to_ingredient, config.data_dir / "idx_to_ingredient.json")
        
        # Save preprocessing metadata
        metadata = {
            "n_recipes": len(train_df) + len(val_df) + len(test_df),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_ingredients": len(self.ingredient_to_idx),
            "n_cuisines": len(self.cuisine_encoder.classes_),
            "text_embedding_dim": text_embeddings.shape[1],
            "ingredient_embedding_dim": ingredient_embeddings.shape[1],
            "graph_nodes": graph_data.num_nodes,
            "graph_edges": graph_data.num_edges,
            "cuisines": list(self.cuisine_encoder.classes_),
            "config_file": str(config.config_path),
            "random_seed": config.random_seed,
        }
        
        save_json(metadata, config.data_dir / "preprocessing_metadata.json")
        
        logging.info("Preprocessed data saved successfully!")
        logging.info(f"Metadata: {metadata}")


def load_raw_data() -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
    """Load raw data from data fetching step."""
    logging.info("Loading raw data...")
    
    # Load recipes
    recipes_path = config.data_dir / "recipes.csv"
    if not recipes_path.exists():
        raise FileNotFoundError(f"Recipes file not found: {recipes_path}")
    
    recipes_df = pd.read_csv(recipes_path)
    logging.info(f"Loaded {len(recipes_df)} recipes")
    
    # Load flavor graph
    flavor_graph_path = config.data_dir / "flavor_graph.json"
    if not flavor_graph_path.exists():
        raise FileNotFoundError(f"Flavor graph file not found: {flavor_graph_path}")
    
    flavor_graph = load_json(flavor_graph_path)
    logging.info(f"Loaded flavor graph with {len(flavor_graph)} nodes")
    
    # Load ratings
    ratings_path = config.data_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    
    ratings_df = pd.read_csv(ratings_path)
    logging.info(f"Loaded {len(ratings_df)} ratings")
    
    return recipes_df, flavor_graph, ratings_df


def main() -> None:
    """Main preprocessing function."""
    setup_logging(config.log_level)
    logging.info("Starting data preprocessing...")
    logging.info(f"Configuration: {config.config_path}")
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    # Load raw data
    recipes_df, flavor_graph, ratings_df = load_raw_data()
    
    # Initialize preprocessor
    preprocessor = RecipePreprocessor()
    
    # Process text features
    text_embeddings = preprocessor.process_text_features(recipes_df)
    
    # Process ingredient features
    ingredient_embeddings, recipe_ingredient_matrix = preprocessor.process_ingredient_features(
        recipes_df, flavor_graph
    )
    
    # Create graph data
    graph_data = preprocessor.create_graph_data(flavor_graph, ingredient_embeddings)
    
    # Encode categorical features
    processed_df = preprocessor.encode_categorical_features(recipes_df)
    
    # Create dataset splits
    train_df, val_df, test_df = preprocessor.create_dataset_splits(processed_df)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(
        train_df, val_df, test_df,
        text_embeddings, ingredient_embeddings, recipe_ingredient_matrix,
        graph_data
    )
    
    logging.info("Data preprocessing completed successfully!")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"📊 Total recipes: {len(processed_df)}")
    print(f"🎯 Training set: {len(train_df)}")
    print(f"✅ Validation set: {len(val_df)}")
    print(f"🧪 Test set: {len(test_df)}")
    print(f"📝 Text embedding dim: {text_embeddings.shape[1]}")
    print(f"🥘 Ingredient embedding dim: {ingredient_embeddings.shape[1]}")
    print(f"🕸️  Graph nodes: {graph_data.num_nodes}")
    print(f"🔗 Graph edges: {graph_data.num_edges}")
    print(f"🍽️  Cuisines: {list(preprocessor.cuisine_encoder.classes_)}")
    print("="*50)


if __name__ == "__main__":
    main()