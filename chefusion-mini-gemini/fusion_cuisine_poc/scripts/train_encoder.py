import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import jsonlines

class RecipeDataset(Dataset):
    def __init__(self, recipes_path, ingredient_map_path):
        self.recipes = []
        with jsonlines.open(recipes_path) as reader:
            for obj in reader:
                self.recipes.append(obj)
        with open(ingredient_map_path) as f:
            self.ingredient_map = json.load(f)
        self.vocab_size = len(self.ingredient_map)

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        recipe = self.recipes[idx]
        bow = torch.zeros(self.vocab_size)
        for ing_idx in recipe["ingredients"]:
            bow[ing_idx] = 1
        return bow, recipe["culture"]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    data_dir = "data"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    dataset = RecipeDataset(os.path.join(data_dir, "recipes_processed.jsonl"), os.path.join(data_dir, "ingredient_map.json"))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = TextEncoder(dataset.vocab_size, 256)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Training text encoder...")
    for epoch in range(5):
        for bow, culture in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            output = model(bow)
            loss = criterion(output, culture)
            loss.backward()
            optimizer.step()

    # Save model
    torch.save(model.state_dict(), os.path.join(models_dir, "text_encoder.pt"))
    print("Text encoder training complete.")

if __name__ == "__main__":
    main()
