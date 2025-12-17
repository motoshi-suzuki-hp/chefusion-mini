#!/usr/bin/env python3
"""Data fetching script for RecipeNLG dataset with OFFLINE_MODE support."""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.utils import clean_recipe_text, resize_image, setup_logging


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """Download file from URL to filepath."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        return True
    except Exception as e:
        logging.warning(f"Failed to download {url}: {e}")
        return False


def create_dummy_image(size: int, color: Tuple[int, int, int]) -> Image.Image:
    """Create a dummy colored image."""
    # Add some texture to make it more realistic
    image = Image.new("RGB", (size, size), color)
    
    # Add some noise
    pixels = np.array(image)
    noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(pixels)


def get_cuisine_ingredients() -> Dict[str, List[str]]:
    """Get ingredient lists for different cuisines."""
    return {
        "japanese": [
            "soy sauce", "miso paste", "rice", "nori", "salmon", "tuna", "wasabi", 
            "ginger", "shiitake mushrooms", "tofu", "sake", "mirin", "sesame oil", 
            "dashi", "green onions", "cucumber", "avocado", "sesame seeds", 
            "tempura batter", "udon noodles", "ramen noodles", "panko breadcrumbs",
            "teriyaki sauce", "rice vinegar", "pickled ginger", "edamame", "seaweed",
            "bonito flakes", "yuzu", "shichimi togarashi", "karashi", "daikon radish",
            "lotus root", "bamboo shoots", "enoki mushrooms", "mochi", "azuki beans",
            "matcha", "shoyu", "katsuobushi", "kombu", "wakame", "mentaiko"
        ],
        "italian": [
            "olive oil", "garlic", "tomatoes", "basil", "mozzarella", "parmesan",
            "pasta", "prosciutto", "balsamic vinegar", "oregano", "rosemary",
            "pine nuts", "sun-dried tomatoes", "ricotta", "pancetta", "arborio rice",
            "white wine", "black pepper", "parsley", "lemon", "capers", "anchovies",
            "pecorino romano", "gorgonzola", "arugula", "mascarpone", "chianti",
            "bresaola", "mortadella", "salami", "focaccia", "ciabatta", "pesto",
            "marinara sauce", "cannellini beans", "porcini mushrooms", "truffle oil",
            "buffalo mozzarella", "san marzano tomatoes", "guanciale", "nduja"
        ]
    }


def generate_recipe_text(cuisine: str, ingredients: List[str]) -> Tuple[str, str]:
    """Generate realistic recipe title and instructions."""
    cuisine_styles = {
        "japanese": {
            "prefixes": ["Traditional", "Homestyle", "Classic", "Authentic", "Modern"],
            "techniques": ["grilled", "steamed", "tempura", "sashimi", "teriyaki", "miso-glazed"],
            "descriptors": ["with dashi broth", "served with rice", "in soy-based sauce", "with wasabi", "tempura style"]
        },
        "italian": {
            "prefixes": ["Classic", "Traditional", "Rustic", "Authentic", "Homemade"],
            "techniques": ["pasta", "risotto", "pizza", "osso buco", "carbonara", "marinara"],
            "descriptors": ["with olive oil", "in tomato sauce", "with parmesan", "al dente", "rustic style"]
        }
    }
    
    style = cuisine_styles[cuisine]
    main_ingredient = random.choice(ingredients)
    
    # Generate title
    title = f"{random.choice(style['prefixes'])} {random.choice(style['techniques'])} with {main_ingredient}"
    
    # Generate instructions
    instructions_parts = [
        f"Heat {random.choice(['olive oil', 'oil', 'butter'])} in a {random.choice(['pan', 'skillet', 'pot'])}.",
        f"Add {', '.join(ingredients[:3])} and cook for {random.randint(3, 10)} minutes.",
        f"Season with salt and pepper to taste.",
        f"Add {', '.join(ingredients[3:5])} and simmer for {random.randint(5, 15)} minutes.",
        f"Serve {random.choice(style['descriptors'])}.",
        f"Garnish with {random.choice(['fresh herbs', 'lemon', 'parmesan', 'sesame seeds'])}."
    ]
    
    instructions = " ".join(instructions_parts)
    
    return title, instructions


def create_mock_recipenlg_data() -> pd.DataFrame:
    """Create mock RecipeNLG data for testing purposes."""
    logging.info("Creating mock RecipeNLG data...")
    
    cuisine_ingredients = get_cuisine_ingredients()
    all_recipes = []
    
    # Generate recipes for each cuisine
    for cuisine in config.target_cuisines:
        logging.info(f"Generating {config.target_recipes_per_cuisine} {cuisine} recipes...")
        
        available_ingredients = cuisine_ingredients[cuisine]
        
        for i in tqdm(range(config.target_recipes_per_cuisine), desc=f"Creating {cuisine} recipes"):
            # Select random ingredients (3-8 ingredients per recipe)
            num_ingredients = random.randint(3, 8)
            recipe_ingredients = random.sample(available_ingredients, k=num_ingredients)
            
            # Generate recipe text
            title, instructions = generate_recipe_text(cuisine, recipe_ingredients)
            
            recipe = {
                "id": f"{cuisine[:2]}_{i:06d}",
                "title": title,
                "ingredients": ", ".join(recipe_ingredients),
                "instructions": instructions,
                "cuisine": cuisine,
                "rating": round(random.uniform(2.5, 5.0), 1),
                "image_path": f"{cuisine[:2]}_{i:06d}.jpg",
                "num_ingredients": len(recipe_ingredients),
                "prep_time": random.randint(15, 120),  # minutes
                "cook_time": random.randint(10, 180),  # minutes
            }
            all_recipes.append(recipe)
    
    return pd.DataFrame(all_recipes)


def create_images(recipes_df: pd.DataFrame) -> None:
    """Create recipe images (dummy if OFFLINE_MODE, otherwise attempt download)."""
    images_dir = config.data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if config.offline_mode:
        logging.info("OFFLINE_MODE enabled: Creating dummy images...")
        create_dummy_images(recipes_df, images_dir)
    else:
        logging.info("OFFLINE_MODE disabled: Attempting to create realistic images...")
        # In a real implementation, this would use Unsplash API or similar
        # For now, we'll still create dummy images but with more variety
        create_dummy_images(recipes_df, images_dir)


def create_dummy_images(recipes_df: pd.DataFrame, images_dir: Path) -> None:
    """Create dummy images for recipes."""
    logging.info("Creating dummy recipe images...")
    
    # Define safe color palettes for different cuisines (values within [64, 192])
    color_palettes = {
        "japanese": [(139, 69, 90), (180, 133, 120), (160, 82, 100), (170, 150, 140)],  # Brown tones
        "italian": [(180, 80, 100), (178, 90, 90), (190, 99, 120), (190, 140, 80)]  # Red/orange tones
    }
    
    for _, recipe in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc="Creating images"):
        image_path = images_dir / recipe["image_path"]
        
        if not image_path.exists():
            cuisine = recipe["cuisine"]
            color = random.choice(color_palettes.get(cuisine, [(128, 128, 128)]))
            
            image = create_dummy_image(config.image_size, color)
            image.save(image_path, "JPEG", quality=85)


def create_dummy_image(size: int, base_color: Tuple[int, int, int]) -> Image.Image:
    """Create a dummy recipe image with safe pixel values."""
    # Ensure color values are within safe range [64, 192] to avoid extremes
    base_color = tuple(max(64, min(192, c)) for c in base_color)
    
    # Create image with slight variations to make it look more realistic
    image = Image.new("RGB", (size, size), base_color)
    
    # Add some simple patterns to make images distinguishable
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Add some random shapes with very conservative color variations
    for _ in range(random.randint(2, 5)):
        # Create very small color variations (±15 max) within safe range
        color_var = tuple(
            max(64, min(192, base_color[i] + random.randint(-15, 15))) 
            for i in range(3)
        )
        
        # Random circle or rectangle
        if random.choice([True, False]):
            # Circle
            x, y = random.randint(0, size//2), random.randint(0, size//2)
            r = random.randint(10, size//5)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color_var)
        else:
            # Rectangle
            x1, y1 = random.randint(0, size//2), random.randint(0, size//2)
            x2, y2 = x1 + random.randint(15, size//4), y1 + random.randint(15, size//4)
            draw.rectangle([x1, y1, min(x2, size), min(y2, size)], fill=color_var)
    
    return image


def build_flavor_graph(recipes_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build a flavor graph from ingredient co-occurrences."""
    logging.info("Building flavor graph...")
    
    # Extract all ingredients and count frequencies
    all_ingredients = []
    for ingredients_str in recipes_df["ingredients"]:
        ingredients = [ing.strip().lower() for ing in ingredients_str.split(",")]
        all_ingredients.extend(ingredients)
    
    # Count ingredient frequencies
    ingredient_counts = {}
    for ingredient in all_ingredients:
        ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1
    
    # Filter ingredients by minimum frequency
    frequent_ingredients = [
        ing for ing, count in ingredient_counts.items()
        if count >= config.min_ingredient_frequency
    ]
    
    logging.info(f"Found {len(frequent_ingredients)} frequent ingredients")
    
    # Build adjacency based on co-occurrence
    flavor_graph = {}
    
    for ingredient in frequent_ingredients:
        flavor_graph[ingredient] = []
        
        # Find ingredients that co-occur with this ingredient
        for _, recipe in recipes_df.iterrows():
            recipe_ingredients = [ing.strip().lower() for ing in recipe["ingredients"].split(",")]
            if ingredient in recipe_ingredients:
                # Add other ingredients in the same recipe as connections
                for other_ingredient in recipe_ingredients:
                    if (other_ingredient != ingredient and 
                        other_ingredient in frequent_ingredients and 
                        other_ingredient not in flavor_graph[ingredient]):
                        flavor_graph[ingredient].append(other_ingredient)
    
    return flavor_graph


def generate_user_ratings(recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic user ratings for recipes."""
    logging.info("Generating user ratings...")
    
    num_users = 1000
    ratings_per_user = random.randint(5, 20)
    
    ratings_data = []
    
    for user_id in tqdm(range(num_users), desc="Generating ratings"):
        # Each user rates a random subset of recipes
        user_recipes = recipes_df.sample(n=min(ratings_per_user, len(recipes_df)))
        
        for _, recipe in user_recipes.iterrows():
            # Generate rating with some bias towards the recipe's base rating
            base_rating = recipe["rating"]
            user_rating = np.clip(
                np.random.normal(base_rating, 0.8), 1.0, 5.0
            )
            
            ratings_data.append({
                "user_id": user_id,
                "recipe_id": recipe["id"],
                "rating": round(user_rating, 1),
                "cuisine": recipe["cuisine"]
            })
    
    return pd.DataFrame(ratings_data)


def save_data_summary(recipes_df: pd.DataFrame, flavor_graph: Dict[str, List[str]], 
                     ratings_df: pd.DataFrame) -> None:
    """Save summary statistics about the generated data."""
    summary = {
        "total_recipes": len(recipes_df),
        "recipes_by_cuisine": recipes_df["cuisine"].value_counts().to_dict(),
        "total_images": len(list((config.data_dir / "images").glob("*.jpg"))),
        "flavor_graph_nodes": len(flavor_graph),
        "flavor_graph_edges": sum(len(connections) for connections in flavor_graph.values()),
        "total_ratings": len(ratings_df),
        "avg_rating": float(recipes_df["rating"].mean()),
        "rating_distribution": ratings_df["rating"].value_counts().to_dict(),
        "avg_ingredients_per_recipe": float(recipes_df["num_ingredients"].mean()),
        "offline_mode": config.offline_mode,
    }
    
    summary_path = config.data_dir / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Data summary saved to {summary_path}")
    return summary


def main() -> None:
    """Main data fetching function."""
    setup_logging(config.log_level)
    logging.info("Starting data fetching process...")
    logging.info(f"Configuration: {config.config_path}")
    logging.info(f"Offline mode: {config.offline_mode}")
    
    # Set random seed for reproducibility
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Create data directory
    config.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mock RecipeNLG data
    recipes_df = create_mock_recipenlg_data()
    
    # Clean recipe text
    recipes_df["ingredients"] = recipes_df["ingredients"].apply(clean_recipe_text)
    recipes_df["instructions"] = recipes_df["instructions"].apply(clean_recipe_text)
    
    # Save recipes
    recipes_path = config.data_dir / "recipes.csv"
    recipes_df.to_csv(recipes_path, index=False)
    logging.info(f"Saved {len(recipes_df)} recipes to {recipes_path}")
    
    # Create images
    create_images(recipes_df)
    
    # Build flavor graph
    flavor_graph = build_flavor_graph(recipes_df)
    
    # Save flavor graph
    flavor_graph_path = config.data_dir / "flavor_graph.json"
    with open(flavor_graph_path, "w") as f:
        json.dump(flavor_graph, f, indent=2)
    
    logging.info(f"Saved flavor graph with {len(flavor_graph)} nodes to {flavor_graph_path}")
    
    # Generate user ratings
    ratings_df = generate_user_ratings(recipes_df)
    
    # Save ratings
    ratings_path = config.data_dir / "ratings.csv"
    ratings_df.to_csv(ratings_path, index=False)
    logging.info(f"Saved {len(ratings_df)} ratings to {ratings_path}")
    
    # Save summary
    summary = save_data_summary(recipes_df, flavor_graph, ratings_df)
    
    logging.info("Data fetching completed successfully!")
    logging.info(f"Summary: {summary}")
    
    # Print key statistics
    print("\n" + "="*50)
    print("DATA GENERATION COMPLETE")
    print("="*50)
    print(f"📊 Total recipes: {summary['total_recipes']}")
    print(f"🍣 Japanese recipes: {summary['recipes_by_cuisine']['japanese']}")
    print(f"🍝 Italian recipes: {summary['recipes_by_cuisine']['italian']}")
    print(f"🖼️  Total images: {summary['total_images']}")
    print(f"🕸️  Flavor graph nodes: {summary['flavor_graph_nodes']}")
    print(f"⭐ Total ratings: {summary['total_ratings']}")
    print(f"📈 Average rating: {summary['avg_rating']:.2f}")
    print(f"🔧 Offline mode: {summary['offline_mode']}")
    print("="*50)


if __name__ == "__main__":
    main()