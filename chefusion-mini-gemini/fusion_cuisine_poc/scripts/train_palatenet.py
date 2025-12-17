import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import jsonlines
import networkx as nx
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data

class PalateNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.sage = GraphSAGE(in_channels, hidden_channels, num_layers=2)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        return self.fc(x)

class RatingDataset(Dataset):
    def __init__(self, ratings_path):
        self.ratings = []
        with jsonlines.open(ratings_path) as reader:
            for obj in reader:
                self.ratings.append(obj)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        rating = self.ratings[idx]
        return rating["recipe_id"], rating["rating"]

def main():
    data_dir = "data"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    graph = nx.read_edgelist(os.path.join(data_dir, "flavorgraph.edgelist"))
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.randn(graph.number_of_nodes(), 64)
    data = Data(x=x, edge_index=edge_index)

    dataset = RatingDataset(os.path.join(data_dir, "ratings_processed.jsonl"))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = PalateNet(64, 128, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    print("Training PalateNet...")
    for epoch in range(5):
        for _, rating in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = criterion(output.squeeze(), rating.float())
            loss.backward()
            optimizer.step()

    # Save model
    torch.save(model.state_dict(), os.path.join(models_dir, "palatenet.pt"))
    print("PalateNet training complete.")

if __name__ == "__main__":
    main()
