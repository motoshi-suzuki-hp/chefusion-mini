import json
import os
import random
import pandas as pd
import requests
from tqdm import tqdm

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    recipes_json_path = os.path.join(data_dir, "recipes.json")
    ratings_json_path = os.path.join(data_dir, "ratings.json")

    if not os.path.exists(recipes_json_path) or not os.path.exists(ratings_json_path):
        print("Generating dummy recipes and ratings...")
        combined_recipes = []
        for i in range(5000):
            combined_recipes.append({"id": f"jp_recipe_{i}", "RecipeName": f"Japanese Dish {i}", "title": f"Japanese Dish {i}", "Ingredients": "rice, soy sauce, fish", "cuisine": "japanese", "ingredients": [{"text": "rice"}, {"text": "soy sauce"}, {"text": "fish"}]})
            combined_recipes.append({"id": f"it_recipe_{i}", "RecipeName": f"Italian Dish {i}", "title": f"Italian Dish {i}", "Ingredients": "pasta, tomato, cheese", "cuisine": "italian", "ingredients": [{"text": "pasta"}, {"text": "tomato"}, {"text": "cheese"}]})
        
        with open(recipes_json_path, "w") as f:
            json.dump(combined_recipes, f)

        ratings = []
        for recipe in combined_recipes:
            ratings.append({"recipe_id": recipe["id"], "rating": random.randint(1, 5)})
        with open(ratings_json_path, "w") as f:
            json.dump(ratings, f)
        print("Dummy data generated.")
    else:
        print("Recipes and ratings already exist. Skipping dummy data generation.")

    print("Data fetching complete.")

if __name__ == "__main__":
    main()
