import json
import os
import re
from collections import Counter

import networkx as nx
import jsonlines
from tqdm import tqdm


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def main():
    data_dir = "data"
    print("Starting preprocessing...")

    # Load data
    with open(os.path.join(data_dir, "recipes.json")) as f:
        recipes = json.load(f)
    with open(os.path.join(data_dir, "ratings.json")) as f:
        ratings = json.load(f)

    # Build ingredient counter
    ingredient_counts = Counter()
    for recipe in tqdm(recipes, desc="Counting ingredients"):
        for ingredient in recipe["ingredients"]:
            ingredient_counts[clean_text(ingredient["text"])] += 1

    # Create ingredient map
    ingredient_map = {ing: i for i, (ing, count) in enumerate(ingredient_counts.items()) if count >= 10}
    with open(os.path.join(data_dir, "ingredient_map.json"), "w") as f:
        json.dump(ingredient_map, f)

    # Process recipes and build graph
    processed_recipes = []
    graph = nx.Graph()
    for recipe in tqdm(recipes, desc="Processing recipes and building graph"):
        tokenized_ingredients = [ingredient_map[clean_text(ing["text"])] for ing in recipe["ingredients"] if clean_text(ing["text"]) in ingredient_map]
        if not tokenized_ingredients:
            continue

        processed_recipes.append({
            "id": recipe["id"],
            "title": recipe["title"],
            "ingredients": tokenized_ingredients,
            "culture": 0 if "japanese" in recipe["cuisine"].lower() else 1
        })

        # Add edges to graph
        for i in range(len(tokenized_ingredients)):
            for j in range(i + 1, len(tokenized_ingredients)):
                graph.add_edge(tokenized_ingredients[i], tokenized_ingredients[j])

    # Save processed data
    with jsonlines.open(os.path.join(data_dir, "recipes_processed.jsonl"), "w") as writer:
        writer.write_all(processed_recipes)

    nx.write_edgelist(graph, os.path.join(data_dir, "flavorgraph.edgelist"))

    # Process ratings
    recipe_ids = {recipe["id"] for recipe in processed_recipes}
    processed_ratings = [rating for rating in ratings if rating["recipe_id"] in recipe_ids]
    with jsonlines.open(os.path.join(data_dir, "ratings_processed.jsonl"), "w") as writer:
        writer.write_all(processed_ratings)

    print(f"Preprocessing complete. Saved {len(processed_recipes)} recipes.")
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

if __name__ == "__main__":
    main()
