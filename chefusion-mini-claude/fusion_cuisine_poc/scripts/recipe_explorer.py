#!/usr/bin/env python3
"""
Interactive Recipe Explorer for the Fusion Cuisine project.

This script provides a command-line interface to browse and explore
generated recipes, both original and fusion recipes.

Usage:
    python scripts/recipe_explorer.py [--cuisine japanese|italian] [--count 10] [--random]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import setup_logging


def load_original_recipes(cuisine=None, count=10, random_sample=False):
    """Load original recipes from the dataset."""
    recipes_path = Path("app/data/recipes.csv")
    
    if not recipes_path.exists():
        print("❌ Recipe data not found. Run data generation first.")
        return []
    
    df = pd.read_csv(recipes_path)
    
    # Filter by cuisine if specified
    if cuisine:
        df = df[df["cuisine"] == cuisine.lower()]
    
    # Sample recipes
    if random_sample:
        df = df.sample(n=min(count, len(df)), random_state=42)
    else:
        df = df.head(count)
    
    recipes = []
    for _, row in df.iterrows():
        recipe = {
            "type": "original",
            "cuisine": row["cuisine"],
            "title": row["title"],
            "rating": row["rating"],
            "ingredients": row["ingredients"],
            "instructions": row["instructions"]
        }
        if "num_ingredients" in row:
            recipe["num_ingredients"] = row["num_ingredients"]
        recipes.append(recipe)
    
    return recipes


def load_fusion_recipes():
    """Load fusion recipes if available."""
    fusion_path = Path("outputs/recipes/fusion_recipes.json")
    
    if not fusion_path.exists():
        print("❌ Fusion recipes not found. Run fusion generation first.")
        return []
    
    with open(fusion_path) as f:
        fusion_data = json.load(f)
    
    recipes = []
    for alpha_key, recipe_data in fusion_data.items():
        recipe = {
            "type": "fusion",
            "alpha": recipe_data["alpha"],
            "japanese_weight": recipe_data["alpha"],
            "italian_weight": 1 - recipe_data["alpha"],
            "recipe_text": recipe_data["recipe"]
        }
        recipes.append(recipe)
    
    return recipes


def display_recipe(recipe, index=None):
    """Display a single recipe in formatted output."""
    print("\n" + "=" * 80)
    
    if recipe["type"] == "original":
        title = f"🍽️ {recipe['cuisine'].upper()} RECIPE"
        if index is not None:
            title += f" #{index + 1}"
        print(title)
        print("=" * 80)
        print(f"📝 Title: {recipe['title']}")
        print(f"⭐ Rating: {recipe['rating']:.1f}/5.0")
        if "num_ingredients" in recipe:
            print(f"🥘 Ingredients count: {recipe['num_ingredients']}")
        
        print(f"\n📋 INGREDIENTS:")
        print("-" * 40)
        ingredients = recipe['ingredients']
        if len(ingredients) > 300:
            print(ingredients[:300] + "...")
            print(f"[... truncated, full length: {len(ingredients)} characters]")
        else:
            print(ingredients)
        
        print(f"\n👨‍🍳 INSTRUCTIONS:")
        print("-" * 40)
        instructions = recipe['instructions']
        if len(instructions) > 500:
            print(instructions[:500] + "...")
            print(f"[... truncated, full length: {len(instructions)} characters]")
        else:
            print(instructions)
    
    elif recipe["type"] == "fusion":
        print(f"🌏 FUSION RECIPE (α = {recipe['alpha']:.1f})")
        print("=" * 80)
        print(f"🍣 Japanese influence: {recipe['japanese_weight']:.1f} ({recipe['japanese_weight']*100:.0f}%)")
        print(f"🍝 Italian influence: {recipe['italian_weight']:.1f} ({recipe['italian_weight']*100:.0f}%)")
        
        print(f"\n📝 RECIPE:")
        print("-" * 40)
        recipe_text = recipe['recipe_text']
        print(recipe_text)
    
    print("=" * 80)


def display_recipe_summary(recipes):
    """Display a summary of loaded recipes."""
    if not recipes:
        print("❌ No recipes found.")
        return
    
    print(f"\n📊 RECIPE SUMMARY:")
    print("-" * 30)
    print(f"Total recipes: {len(recipes)}")
    
    # Count by type
    original_count = sum(1 for r in recipes if r["type"] == "original")
    fusion_count = sum(1 for r in recipes if r["type"] == "fusion")
    
    if original_count > 0:
        print(f"Original recipes: {original_count}")
        
        # Count by cuisine for original recipes
        cuisines = {}
        for recipe in recipes:
            if recipe["type"] == "original":
                cuisine = recipe["cuisine"]
                cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
        
        for cuisine, count in cuisines.items():
            print(f"  {cuisine.capitalize()}: {count}")
    
    if fusion_count > 0:
        print(f"Fusion recipes: {fusion_count}")
        
        # Show alpha values for fusion recipes
        alphas = [r["alpha"] for r in recipes if r["type"] == "fusion"]
        print(f"  Alpha values: {', '.join(f'{a:.1f}' for a in sorted(set(alphas)))}")


def browse_recipes_interactive(recipes):
    """Provide interactive browsing of recipes."""
    if not recipes:
        print("❌ No recipes to browse.")
        return
    
    current_index = 0
    
    print(f"\n🔍 INTERACTIVE RECIPE BROWSER")
    print(f"Loaded {len(recipes)} recipes. Use commands to navigate:")
    print("Commands: [n]ext, [p]revious, [r]andom, [g]oto <num>, [q]uit, [h]elp")
    
    while True:
        # Display current recipe
        display_recipe(recipes[current_index], current_index)
        
        # Show navigation info
        print(f"\n📍 Recipe {current_index + 1} of {len(recipes)}")
        
        try:
            command = input("\n👉 Enter command: ").strip().lower()
            
            if command in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif command in ['n', 'next']:
                current_index = (current_index + 1) % len(recipes)
            
            elif command in ['p', 'prev', 'previous']:
                current_index = (current_index - 1) % len(recipes)
            
            elif command in ['r', 'random']:
                current_index = random.randint(0, len(recipes) - 1)
            
            elif command.startswith('g'):
                # Parse goto command
                parts = command.split()
                if len(parts) == 2 and parts[1].isdigit():
                    target = int(parts[1]) - 1  # Convert to 0-based index
                    if 0 <= target < len(recipes):
                        current_index = target
                    else:
                        print(f"❌ Invalid recipe number. Use 1-{len(recipes)}")
                else:
                    print("❌ Invalid goto command. Use: g <number>")
            
            elif command in ['h', 'help']:
                print("\n📖 COMMANDS:")
                print("  n, next     - Go to next recipe")
                print("  p, previous - Go to previous recipe") 
                print("  r, random   - Go to random recipe")
                print("  g <num>     - Go to recipe number <num>")
                print("  q, quit     - Exit browser")
                print("  h, help     - Show this help")
            
            else:
                print("❌ Unknown command. Type 'h' for help.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


def search_recipes(recipes, search_term):
    """Search recipes by title or ingredients."""
    if not recipes:
        return []
    
    search_term = search_term.lower()
    matching_recipes = []
    
    for recipe in recipes:
        # Search in title
        title_match = search_term in recipe.get("title", "").lower()
        
        # Search in ingredients
        ingredients_match = search_term in recipe.get("ingredients", "").lower()
        
        # Search in fusion recipe text
        fusion_match = False
        if recipe["type"] == "fusion":
            fusion_match = search_term in recipe.get("recipe_text", "").lower()
        
        if title_match or ingredients_match or fusion_match:
            matching_recipes.append(recipe)
    
    return matching_recipes


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Explore Fusion Cuisine recipes")
    parser.add_argument("--cuisine", choices=["japanese", "italian"], 
                        help="Filter by cuisine (for original recipes)")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of recipes to load (default: 10)")
    parser.add_argument("--random", action="store_true",
                        help="Load random recipes")
    parser.add_argument("--fusion", action="store_true",
                        help="Load fusion recipes instead of original")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start interactive browser")
    parser.add_argument("--search", type=str,
                        help="Search for recipes containing this term")
    parser.add_argument("--export", type=str,
                        help="Export recipes to JSON file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    print("🍽️ FUSION CUISINE RECIPE EXPLORER")
    print("=" * 50)
    
    # Load recipes
    if args.fusion:
        print("Loading fusion recipes...")
        recipes = load_fusion_recipes()
    else:
        print(f"Loading original recipes...")
        if args.cuisine:
            print(f"Filtering by cuisine: {args.cuisine}")
        recipes = load_original_recipes(args.cuisine, args.count, args.random)
    
    if not recipes:
        print("❌ No recipes loaded. Check that data generation completed successfully.")
        return
    
    # Apply search filter if specified
    if args.search:
        print(f"Searching for: '{args.search}'...")
        recipes = search_recipes(recipes, args.search)
        print(f"Found {len(recipes)} matching recipes.")
    
    # Display summary
    display_recipe_summary(recipes)
    
    # Export if requested
    if args.export:
        with open(args.export, 'w') as f:
            json.dump(recipes, f, indent=2, default=str)
        print(f"✅ Recipes exported to: {args.export}")
    
    # Start interactive mode or display recipes
    if args.interactive:
        browse_recipes_interactive(recipes)
    else:
        # Display all recipes
        for i, recipe in enumerate(recipes):
            display_recipe(recipe, i)
            
            # Pause between recipes if there are many
            if len(recipes) > 5 and i < len(recipes) - 1:
                try:
                    input(f"\nPress Enter to continue to recipe {i + 2}/{len(recipes)} (or Ctrl+C to stop)...")
                except KeyboardInterrupt:
                    print(f"\n\nShowing first {i + 1} of {len(recipes)} recipes.")
                    break
    
    print(f"\n💡 TIP: Use --interactive flag for interactive browsing")
    print(f"💡 TIP: Use --search '<term>' to find specific recipes")


if __name__ == "__main__":
    main()