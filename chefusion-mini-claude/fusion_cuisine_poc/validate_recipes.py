#!/usr/bin/env python3
"""
Recipe Validation Script

This script validates the quality and consistency of generated recipes.

Usage:
    python validate_recipes.py
"""

import pandas as pd
import json
from pathlib import Path


def validate_recipe_quality(df):
    """Validate the quality of generated recipes."""
    print("🔍 RECIPE QUALITY VALIDATION")
    print("=" * 35)
    
    issues = []
    
    # Check for missing data
    missing_titles = df['title'].isna().sum()
    missing_ingredients = df['ingredients'].isna().sum()
    missing_instructions = df['instructions'].isna().sum()
    
    if missing_titles > 0:
        issues.append(f"❌ {missing_titles} recipes missing titles")
    if missing_ingredients > 0:
        issues.append(f"❌ {missing_ingredients} recipes missing ingredients")
    if missing_instructions > 0:
        issues.append(f"❌ {missing_instructions} recipes missing instructions")
    
    # Check for empty content
    empty_titles = (df['title'].str.len() < 5).sum()
    empty_ingredients = (df['ingredients'].str.len() < 10).sum()
    empty_instructions = (df['instructions'].str.len() < 20).sum()
    
    if empty_titles > 0:
        issues.append(f"⚠️ {empty_titles} recipes with very short titles")
    if empty_ingredients > 0:
        issues.append(f"⚠️ {empty_ingredients} recipes with very short ingredient lists")
    if empty_instructions > 0:
        issues.append(f"⚠️ {empty_instructions} recipes with very short instructions")
    
    # Check rating range
    invalid_ratings = ((df['rating'] < 1.0) | (df['rating'] > 5.0)).sum()
    if invalid_ratings > 0:
        issues.append(f"❌ {invalid_ratings} recipes with invalid ratings")
    
    # Check cuisine values
    valid_cuisines = ['japanese', 'italian']
    invalid_cuisines = (~df['cuisine'].isin(valid_cuisines)).sum()
    if invalid_cuisines > 0:
        issues.append(f"❌ {invalid_cuisines} recipes with invalid cuisine types")
    
    # Report results
    if not issues:
        print("✅ All validation checks passed!")
        print("🎉 Recipe quality is excellent!")
    else:
        print("⚠️ Validation issues found:")
        for issue in issues:
            print(f"   {issue}")
    
    print()
    return len(issues) == 0


def analyze_recipe_diversity(df):
    """Analyze the diversity of generated recipes."""
    print("🌈 RECIPE DIVERSITY ANALYSIS")
    print("=" * 30)
    
    # Title diversity
    unique_titles = df['title'].nunique()
    title_diversity = (unique_titles / len(df)) * 100
    print(f"📝 Title diversity: {unique_titles:,} unique titles ({title_diversity:.1f}%)")
    
    # Check for duplicates
    duplicate_titles = df['title'].duplicated().sum()
    if duplicate_titles > 0:
        print(f"⚠️ Found {duplicate_titles} duplicate titles")
    else:
        print("✅ All titles are unique")
    
    # Ingredient analysis
    all_ingredients = []
    for ingredients_str in df['ingredients']:
        ingredients_list = [ing.strip() for ing in ingredients_str.split(',')]
        all_ingredients.extend(ingredients_list)
    
    unique_ingredients = len(set(all_ingredients))
    print(f"🥘 Unique ingredients: {unique_ingredients}")
    
    # Most common ingredients
    from collections import Counter
    ingredient_counts = Counter(all_ingredients)
    print(f"🔝 Top 10 ingredients:")
    for ingredient, count in ingredient_counts.most_common(10):
        percentage = (count / len(all_ingredients)) * 100
        print(f"   {ingredient}: {count} uses ({percentage:.1f}%)")
    
    print()


def show_recipe_samples(df, count=5):
    """Show detailed recipe samples."""
    print(f"📖 DETAILED RECIPE SAMPLES ({count} per cuisine)")
    print("=" * 45)
    
    for cuisine in df['cuisine'].unique():
        cuisine_df = df[df['cuisine'] == cuisine]
        samples = cuisine_df.head(count)
        
        print(f"\n🎌 {cuisine.upper()} RECIPES:")
        print("-" * 25)
        
        for i, (_, recipe) in enumerate(samples.iterrows(), 1):
            print(f"\n📝 Recipe #{i}: {recipe['title']}")
            print(f"⭐ Rating: {recipe['rating']:.1f}/5.0")
            
            if 'num_ingredients' in recipe:
                print(f"🥘 Ingredient count: {recipe['num_ingredients']}")
            
            print(f"\n🛒 Ingredients:")
            ingredients = recipe['ingredients']
            ingredient_list = [ing.strip() for ing in ingredients.split(',')]
            for j, ingredient in enumerate(ingredient_list[:8], 1):  # Show first 8
                print(f"   {j}. {ingredient}")
            if len(ingredient_list) > 8:
                print(f"   ... and {len(ingredient_list) - 8} more")
            
            print(f"\n👨‍🍳 Instructions:")
            instructions = recipe['instructions']
            # Split into sentences for better readability
            sentences = instructions.split('. ')
            for sentence in sentences[:3]:  # Show first 3 sentences
                if sentence.strip():
                    print(f"   • {sentence.strip()}{'.' if not sentence.endswith('.') else ''}")
            if len(sentences) > 3:
                print("   • ...")
            
            print("-" * 40)


def generate_recipe_report(df):
    """Generate a comprehensive recipe report."""
    print("\n📊 COMPREHENSIVE RECIPE REPORT")
    print("=" * 40)
    
    total_recipes = len(df)
    
    # Basic statistics
    print(f"📈 Dataset Statistics:")
    print(f"   Total recipes: {total_recipes:,}")
    print(f"   Average rating: {df['rating'].mean():.2f}")
    print(f"   Rating std dev: {df['rating'].std():.2f}")
    
    if 'num_ingredients' in df.columns:
        print(f"   Avg ingredients per recipe: {df['num_ingredients'].mean():.1f}")
    
    # Cuisine breakdown
    print(f"\n🌍 Cuisine Breakdown:")
    for cuisine in df['cuisine'].unique():
        count = (df['cuisine'] == cuisine).sum()
        avg_rating = df[df['cuisine'] == cuisine]['rating'].mean()
        print(f"   {cuisine.capitalize()}: {count:,} recipes (avg rating: {avg_rating:.2f})")
    
    # Rating distribution
    print(f"\n⭐ Rating Distribution:")
    rating_ranges = [
        (1.0, 2.0, "Poor"),
        (2.0, 3.0, "Fair"), 
        (3.0, 4.0, "Good"),
        (4.0, 5.0, "Excellent")
    ]
    
    for min_rating, max_rating, label in rating_ranges:
        count = ((df['rating'] >= min_rating) & (df['rating'] < max_rating)).sum()
        percentage = (count / total_recipes) * 100
        print(f"   {label} ({min_rating:.1f}-{max_rating:.1f}): {count:,} ({percentage:.1f}%)")
    
    # Perfect ratings
    perfect_count = (df['rating'] == 5.0).sum()
    perfect_percentage = (perfect_count / total_recipes) * 100
    print(f"   Perfect (5.0): {perfect_count:,} ({perfect_percentage:.1f}%)")


def main():
    """Main validation function."""
    recipes_path = Path("app/data/recipes.csv")
    
    if not recipes_path.exists():
        print("❌ Recipe file not found!")
        print("Please run: python confirm_recipes.py first")
        return
    
    # Load recipes
    df = pd.read_csv(recipes_path)
    print(f"📖 Loaded {len(df):,} recipes for validation\n")
    
    # Run validations
    is_valid = validate_recipe_quality(df)
    analyze_recipe_diversity(df)
    show_recipe_samples(df, count=3)
    generate_recipe_report(df)
    
    # Final summary
    print(f"\n🎯 VALIDATION SUMMARY")
    print("=" * 25)
    if is_valid:
        print("✅ Recipe validation PASSED")
        print("🎉 Recipes are ready for use!")
    else:
        print("⚠️ Recipe validation found issues")
        print("🔧 Consider regenerating data")
    
    print(f"\n💾 Recipe file: {recipes_path}")
    print("🔍 Use recipe_explorer.py for interactive browsing")


if __name__ == "__main__":
    main()