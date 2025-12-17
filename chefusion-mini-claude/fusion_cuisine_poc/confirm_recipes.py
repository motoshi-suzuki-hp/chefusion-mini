#!/usr/bin/env python3
"""
Recipe Confirmation Script

This script helps you confirm and validate the generated recipes
in the Fusion Cuisine project.

Usage:
    python confirm_recipes.py
"""

import pandas as pd
import json
from pathlib import Path


def confirm_recipes():
    """Confirm the generated recipes."""
    print("🍽️ RECIPE CONFIRMATION")
    print("=" * 40)
    
    recipes_path = Path("app/data/recipes.csv")
    
    if not recipes_path.exists():
        print("❌ Recipe file not found at app/data/recipes.csv")
        print("Please run data generation first: make data")
        return
    
    # Load recipes
    df = pd.read_csv(recipes_path)
    
    print(f"✅ Loaded {len(df):,} recipes")
    print(f"📊 Columns: {', '.join(df.columns)}")
    print()
    
    # Cuisine distribution
    print("🌍 CUISINE DISTRIBUTION:")
    cuisine_counts = df['cuisine'].value_counts()
    for cuisine, count in cuisine_counts.items():
        emoji = "🍣" if cuisine == "japanese" else "🍝" if cuisine == "italian" else "🍽️"
        percentage = (count / len(df)) * 100
        print(f"   {emoji} {cuisine.capitalize()}: {count:,} ({percentage:.1f}%)")
    
    print()
    
    # Rating statistics
    print("⭐ RATING STATISTICS:")
    print(f"   Average: {df['rating'].mean():.2f}")
    print(f"   Range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
    print(f"   Standard deviation: {df['rating'].std():.2f}")
    
    print()
    
    # Ingredient statistics
    if 'num_ingredients' in df.columns:
        print("🥘 INGREDIENT STATISTICS:")
        print(f"   Average per recipe: {df['num_ingredients'].mean():.1f}")
        print(f"   Range: {df['num_ingredients'].min()} - {df['num_ingredients'].max()}")
    
    print()
    
    # Sample recipes
    print("📝 SAMPLE RECIPES:")
    print("=" * 30)
    
    for cuisine in df['cuisine'].unique():
        print(f"\n🎌 {cuisine.upper()} SAMPLES:")
        samples = df[df['cuisine'] == cuisine].head(3)
        
        for i, (_, recipe) in enumerate(samples.iterrows(), 1):
            print(f"\n   #{i} {recipe['title']}")
            print(f"       Rating: ⭐{recipe['rating']:.1f}/5.0")
            
            # Show ingredients (truncated)
            ingredients = recipe['ingredients']
            if len(ingredients) > 80:
                print(f"       Ingredients: {ingredients[:80]}...")
            else:
                print(f"       Ingredients: {ingredients}")
            
            # Show instructions (truncated)
            instructions = recipe['instructions']
            if len(instructions) > 100:
                print(f"       Instructions: {instructions[:100]}...")
            else:
                print(f"       Instructions: {instructions}")
    
    print("\n" + "=" * 40)
    print("✅ Recipe confirmation complete!")
    print(f"📁 Full dataset: {recipes_path}")
    print("💡 Use recipe_explorer.py for interactive browsing")


if __name__ == "__main__":
    confirm_recipes()