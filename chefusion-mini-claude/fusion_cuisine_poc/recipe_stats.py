#!/usr/bin/env python3
"""
Recipe Statistics Generator

This script generates detailed statistics about the generated recipes.

Usage:
    python recipe_stats.py
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def generate_basic_stats(df):
    """Generate basic recipe statistics."""
    print("📊 BASIC RECIPE STATISTICS")
    print("=" * 35)
    
    total_recipes = len(df)
    print(f"Total recipes: {total_recipes:,}")
    
    # Cuisine distribution
    print(f"\n🌍 Cuisine Distribution:")
    cuisine_counts = df['cuisine'].value_counts()
    for cuisine, count in cuisine_counts.items():
        percentage = (count / total_recipes) * 100
        emoji = "🍣" if cuisine == "japanese" else "🍝"
        print(f"   {emoji} {cuisine.capitalize()}: {count:,} ({percentage:.1f}%)")
    
    # Rating statistics
    print(f"\n⭐ Rating Statistics:")
    print(f"   Mean: {df['rating'].mean():.3f}")
    print(f"   Median: {df['rating'].median():.3f}")
    print(f"   Std Dev: {df['rating'].std():.3f}")
    print(f"   Min: {df['rating'].min():.1f}")
    print(f"   Max: {df['rating'].max():.1f}")
    
    # Rating distribution by quartiles
    quartiles = df['rating'].quantile([0.25, 0.5, 0.75])
    print(f"   Q1 (25%): {quartiles[0.25]:.2f}")
    print(f"   Q2 (50%): {quartiles[0.5]:.2f}")
    print(f"   Q3 (75%): {quartiles[0.75]:.2f}")


def analyze_ingredients(df):
    """Analyze ingredient usage patterns."""
    print(f"\n🥘 INGREDIENT ANALYSIS")
    print("=" * 25)
    
    # Collect all ingredients
    all_ingredients = []
    for ingredients_str in df['ingredients']:
        ingredients_list = [ing.strip() for ing in ingredients_str.split(',')]
        all_ingredients.extend(ingredients_list)
    
    ingredient_counter = Counter(all_ingredients)
    unique_ingredients = len(ingredient_counter)
    total_ingredient_uses = len(all_ingredients)
    
    print(f"Unique ingredients: {unique_ingredients}")
    print(f"Total ingredient uses: {total_ingredient_uses:,}")
    print(f"Average uses per ingredient: {total_ingredient_uses / unique_ingredients:.1f}")
    
    # Most common ingredients
    print(f"\n🔝 Top 15 Most Used Ingredients:")
    for i, (ingredient, count) in enumerate(ingredient_counter.most_common(15), 1):
        percentage = (count / total_ingredient_uses) * 100
        print(f"   {i:2d}. {ingredient:<20} {count:4d} uses ({percentage:4.1f}%)")
    
    # Ingredients per recipe statistics
    if 'num_ingredients' in df.columns:
        print(f"\n📝 Ingredients per Recipe:")
        print(f"   Mean: {df['num_ingredients'].mean():.1f}")
        print(f"   Median: {df['num_ingredients'].median():.1f}")
        print(f"   Min: {df['num_ingredients'].min()}")
        print(f"   Max: {df['num_ingredients'].max()}")
    
    return ingredient_counter


def analyze_by_cuisine(df):
    """Analyze recipes by cuisine type."""
    print(f"\n🎌 CUISINE-SPECIFIC ANALYSIS")
    print("=" * 30)
    
    for cuisine in df['cuisine'].unique():
        cuisine_df = df[df['cuisine'] == cuisine]
        emoji = "🍣" if cuisine == "japanese" else "🍝"
        
        print(f"\n{emoji} {cuisine.upper()} CUISINE:")
        print(f"   Recipes: {len(cuisine_df):,}")
        print(f"   Avg Rating: {cuisine_df['rating'].mean():.3f}")
        print(f"   Rating Range: {cuisine_df['rating'].min():.1f} - {cuisine_df['rating'].max():.1f}")
        
        if 'num_ingredients' in cuisine_df.columns:
            print(f"   Avg Ingredients: {cuisine_df['num_ingredients'].mean():.1f}")
        
        # Top ingredients for this cuisine
        cuisine_ingredients = []
        for ingredients_str in cuisine_df['ingredients']:
            ingredients_list = [ing.strip() for ing in ingredients_str.split(',')]
            cuisine_ingredients.extend(ingredients_list)
        
        cuisine_counter = Counter(cuisine_ingredients)
        print(f"   Top 5 Ingredients:")
        for i, (ingredient, count) in enumerate(cuisine_counter.most_common(5), 1):
            percentage = (count / len(cuisine_ingredients)) * 100
            print(f"      {i}. {ingredient} ({count} uses, {percentage:.1f}%)")


def analyze_titles(df):
    """Analyze recipe titles."""
    print(f"\n📝 TITLE ANALYSIS")
    print("=" * 20)
    
    # Title length statistics
    title_lengths = df['title'].str.len()
    print(f"Title Length Statistics:")
    print(f"   Mean: {title_lengths.mean():.1f} characters")
    print(f"   Median: {title_lengths.median():.1f} characters")
    print(f"   Min: {title_lengths.min()} characters")
    print(f"   Max: {title_lengths.max()} characters")
    
    # Word count in titles
    word_counts = df['title'].str.split().str.len()
    print(f"\nTitle Word Count:")
    print(f"   Mean: {word_counts.mean():.1f} words")
    print(f"   Median: {word_counts.median():.1f} words")
    print(f"   Min: {word_counts.min()} words")
    print(f"   Max: {word_counts.max()} words")
    
    # Common words in titles
    all_title_words = []
    for title in df['title']:
        words = title.lower().split()
        all_title_words.extend(words)
    
    word_counter = Counter(all_title_words)
    print(f"\n🔤 Most Common Title Words:")
    for i, (word, count) in enumerate(word_counter.most_common(10), 1):
        percentage = (count / len(all_title_words)) * 100
        print(f"   {i:2d}. {word:<15} {count:3d} uses ({percentage:4.1f}%)")


def analyze_instructions(df):
    """Analyze cooking instructions."""
    print(f"\n👨‍🍳 INSTRUCTION ANALYSIS")
    print("=" * 25)
    
    # Instruction length statistics
    instruction_lengths = df['instructions'].str.len()
    print(f"Instruction Length (characters):")
    print(f"   Mean: {instruction_lengths.mean():.0f}")
    print(f"   Median: {instruction_lengths.median():.0f}")
    print(f"   Min: {instruction_lengths.min()}")
    print(f"   Max: {instruction_lengths.max()}")
    
    # Word count in instructions
    instruction_word_counts = df['instructions'].str.split().str.len()
    print(f"\nInstruction Length (words):")
    print(f"   Mean: {instruction_word_counts.mean():.1f}")
    print(f"   Median: {instruction_word_counts.median():.1f}")
    print(f"   Min: {instruction_word_counts.min()}")
    print(f"   Max: {instruction_word_counts.max()}")
    
    # Common cooking verbs
    all_instruction_words = []
    for instruction in df['instructions']:
        words = instruction.lower().split()
        all_instruction_words.extend(words)
    
    instruction_counter = Counter(all_instruction_words)
    
    # Filter for cooking verbs
    cooking_verbs = ['add', 'cook', 'heat', 'serve', 'simmer', 'season', 'garnish', 
                    'mix', 'stir', 'boil', 'fry', 'bake', 'roast', 'grill', 'steam']
    
    print(f"\n🔥 Common Cooking Actions:")
    verb_counts = []
    for verb in cooking_verbs:
        count = instruction_counter.get(verb, 0)
        if count > 0:
            verb_counts.append((verb, count))
    
    verb_counts.sort(key=lambda x: x[1], reverse=True)
    for i, (verb, count) in enumerate(verb_counts[:10], 1):
        percentage = (count / len(all_instruction_words)) * 100
        print(f"   {i:2d}. {verb:<10} {count:3d} uses ({percentage:4.1f}%)")


def generate_summary_report(df, ingredient_counter):
    """Generate a summary report."""
    print(f"\n📋 SUMMARY REPORT")
    print("=" * 20)
    
    total_recipes = len(df)
    unique_ingredients = len(ingredient_counter)
    avg_rating = df['rating'].mean()
    
    print(f"Dataset Overview:")
    print(f"   📊 Total Recipes: {total_recipes:,}")
    print(f"   🥘 Unique Ingredients: {unique_ingredients}")
    print(f"   ⭐ Average Rating: {avg_rating:.2f}/5.0")
    
    # Quality indicators
    high_rated = (df['rating'] >= 4.0).sum()
    low_rated = (df['rating'] <= 2.0).sum()
    
    print(f"\nQuality Distribution:")
    print(f"   🌟 High-rated (≥4.0): {high_rated:,} ({high_rated/total_recipes*100:.1f}%)")
    print(f"   📉 Low-rated (≤2.0): {low_rated:,} ({low_rated/total_recipes*100:.1f}%)")
    
    # Data completeness
    complete_recipes = df.dropna().shape[0]
    completeness = (complete_recipes / total_recipes) * 100
    
    print(f"\nData Quality:")
    print(f"   ✅ Complete recipes: {complete_recipes:,} ({completeness:.1f}%)")
    
    # Diversity metrics
    unique_titles = df['title'].nunique()
    title_diversity = (unique_titles / total_recipes) * 100
    
    print(f"\nDiversity Metrics:")
    print(f"   📝 Unique titles: {unique_titles:,} ({title_diversity:.1f}%)")
    print(f"   🌍 Cuisines: {df['cuisine'].nunique()}")


def main():
    """Main statistics function."""
    recipes_path = Path("app/data/recipes.csv")
    
    if not recipes_path.exists():
        print("❌ Recipe file not found!")
        print("Please run data generation first")
        return
    
    # Load recipes
    df = pd.read_csv(recipes_path)
    
    print("📈 FUSION CUISINE RECIPE STATISTICS")
    print("=" * 45)
    print(f"📁 Analyzing: {recipes_path}")
    print(f"📊 Dataset size: {len(df):,} recipes")
    print()
    
    # Generate all statistics
    generate_basic_stats(df)
    ingredient_counter = analyze_ingredients(df)
    analyze_by_cuisine(df)
    analyze_titles(df)
    analyze_instructions(df)
    generate_summary_report(df, ingredient_counter)
    
    print(f"\n✅ Statistical analysis complete!")
    print(f"💡 Use validate_recipes.py for quality validation")
    print(f"💡 Use export_recipes.py to export data")


if __name__ == "__main__":
    main()