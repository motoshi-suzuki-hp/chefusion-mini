import json
import os

import torch
import openai

from train_encoder import TextEncoder

def main():
    models_dir = "models"
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    # Load encoder
    with open("data/ingredient_map.json") as f:
        ingredient_map = json.load(f)
    model = TextEncoder(len(ingredient_map), 256)
    model.load_state_dict(torch.load(os.path.join(models_dir, "text_encoder.pt")))
    model.eval()

    # Get culture embeddings
    jp_embedding = model(torch.ones(1, len(ingredient_map))) # Dummy input
    it_embedding = model(torch.zeros(1, len(ingredient_map))) # Dummy input

    # Fuse embeddings
    fused_embedding = 0.5 * jp_embedding + 0.5 * it_embedding

    # Generate recipe
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Generate a recipe that is a fusion of Japanese and Italian cuisine. The embedding for this fusion is {fused_embedding.tolist()}",
        max_tokens=256,
    )

    # Save recipe
    with open(os.path.join(outputs_dir, "fused_recipe.txt"), "w") as f:
        f.write(response.choices[0].text)

    print("Recipe generation complete.")

if __name__ == "__main__":
    main()
