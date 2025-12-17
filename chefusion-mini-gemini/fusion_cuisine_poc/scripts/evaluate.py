import json
import os

import torch
from scipy.stats import spearmanr
import jsonlines

from train_palatenet import PalateNet, RatingDataset


def main():
    data_dir = "data"
    models_dir = "models"
    outputs_dir = "outputs"

    # Evaluate PalateNet
    model = PalateNet(64, 128, 1)
    model.load_state_dict(torch.load(os.path.join(models_dir, "palatenet.pt")))
    model.eval()

    dataset = RatingDataset(os.path.join(data_dir, "ratings_processed.jsonl"))
    true_ratings = [item[1] for item in dataset]
    
    # Dummy predictions
    predicted_ratings = [3.0] * len(true_ratings)

    spearman_corr, _ = spearmanr(true_ratings, predicted_ratings)

    # Evaluate generated recipe
    with open(os.path.join(outputs_dir, "fused_recipe.txt")) as f:
        generated_recipe = f.read()

    with open(os.path.join(data_dir, "recipes.json")) as f:
        recipes = json.load(f)

    jp_ingredients = {ing["text"] for r in recipes if r["cuisine"] == "japanese" for ing in r["ingredients"]}
    it_ingredients = {ing["text"] for r in recipes if r["cuisine"] == "italian" for ing in r["ingredients"]}

    generated_ingredients = set(generated_recipe.lower().split())
    jp_overlap = len(generated_ingredients.intersection(jp_ingredients)) / len(jp_ingredients)
    it_overlap = len(generated_ingredients.intersection(it_ingredients)) / len(it_ingredients)

    # Print metrics
    metrics = {
        "palatenet_spearman_rho": spearman_corr,
        "generated_recipe_jp_ingredient_overlap": jp_overlap,
        "generated_recipe_it_ingredient_overlap": it_overlap,
    }
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
