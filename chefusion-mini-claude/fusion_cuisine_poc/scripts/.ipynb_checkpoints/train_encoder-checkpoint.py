#!/usr/bin/env python3
"""Encoder training script - CLIP-mini for text/image fusion."""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.utils import (
    get_device,
    get_model_size,
    load_pickle,
    set_random_seeds,
    setup_logging,
)


class TextEncoder(nn.Module):
    """Lightweight text encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), p=2, dim=1)


class ImageEncoder(nn.Module):
    """Lightweight image encoder."""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class CLIPMini(nn.Module):
    """Lightweight CLIP-style model for text and image encoding."""
    
    def __init__(self, text_input_dim: int, latent_dim: int = 256):
        super().__init__()
        self.text_encoder = TextEncoder(text_input_dim, 512, latent_dim)
        self.image_encoder = ImageEncoder(latent_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, text_features: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embeddings = self.text_encoder(text_features)
        image_embeddings = self.image_encoder(images)
        return text_embeddings, image_embeddings
    
    def compute_loss(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between text and image embeddings."""
        # Compute similarity matrix
        logits = torch.matmul(text_embeddings, image_embeddings.T) * self.temperature.exp()
        
        # Create labels (diagonal should be positive pairs)
        batch_size = text_embeddings.size(0)
        labels = torch.arange(batch_size, device=text_embeddings.device)
        
        # Symmetric loss (text-to-image + image-to-text)
        loss_text_to_image = F.cross_entropy(logits, labels)
        loss_image_to_text = F.cross_entropy(logits.T, labels)
        
        return (loss_text_to_image + loss_image_to_text) / 2


class RecipeImageDataset(Dataset):
    """Dataset for recipe text and images."""
    
    def __init__(self, recipes_df: pd.DataFrame, text_embeddings: np.ndarray, 
                 image_dir: Path, transform: Optional[transforms.Compose] = None):
        self.recipes_df = recipes_df
        self.text_embeddings = text_embeddings
        self.image_dir = image_dir
        self.transform = transform
        
        # Filter out recipes without valid images
        self.valid_indices = []
        for idx, recipe in recipes_df.iterrows():
            image_path = image_dir / recipe["image_path"]
            if image_path.exists():
                self.valid_indices.append(idx)
        
        logging.info(f"Dataset has {len(self.valid_indices)} valid samples out of {len(recipes_df)}")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.valid_indices[idx]
        recipe = self.recipes_df.iloc[real_idx]
        
        # Load text embeddings
        text_features = torch.tensor(self.text_embeddings[real_idx], dtype=torch.float32)
        
        # Load image
        image_path = self.image_dir / recipe["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform if none provided
                image = transforms.ToTensor()(image)
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            # Create dummy image with proper normalization
            image_size = config.get("dataset.image_size", 256)
            image = torch.zeros(3, image_size, image_size)
            # Apply same normalization as transform would
            if self.transform:
                # Create a dummy white image instead of zeros for better compatibility
                dummy_image = Image.new("RGB", (image_size, image_size), (128, 128, 128))
                try:
                    image = self.transform(dummy_image)
                except:
                    # If transform fails, use normalized zeros
                    normalize_mean = config.get("preprocessing.normalize_mean", [0.485, 0.456, 0.406])
                    normalize_std = config.get("preprocessing.normalize_std", [0.229, 0.224, 0.225])
                    image = torch.zeros(3, image_size, image_size)
                    for i in range(3):
                        image[i] = (image[i] - normalize_mean[i]) / normalize_std[i]
        
        return {
            "text_features": text_features,
            "image": image,
            "recipe_id": recipe["id"],
            "cuisine": recipe["cuisine"]
        }


def create_data_transforms() -> transforms.Compose:
    """Create image transforms for training."""
    normalize_mean = config.get("preprocessing.normalize_mean", [0.485, 0.456, 0.406])
    normalize_std = config.get("preprocessing.normalize_std", [0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Reduced ColorJitter to prevent extreme values
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        # Add clamping to ensure values stay in valid range
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


def create_validation_transforms() -> transforms.Compose:
    """Create image transforms for validation."""
    normalize_mean = config.get("preprocessing.normalize_mean", [0.485, 0.456, 0.406])
    normalize_std = config.get("preprocessing.normalize_std", [0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        # Add clamping for validation too
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


def train_epoch(model: CLIPMini, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        text_features = batch["text_features"].to(device)
        images = batch["image"].to(device)
        
        optimizer.zero_grad()
        
        text_embeddings, image_embeddings = model(text_features, images)
        loss = model.compute_loss(text_embeddings, image_embeddings)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model: CLIPMini, dataloader: DataLoader, device: torch.device) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            text_features = batch["text_features"].to(device)
            images = batch["image"].to(device)
            
            text_embeddings, image_embeddings = model(text_features, images)
            loss = model.compute_loss(text_embeddings, image_embeddings)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def save_model(model: CLIPMini, optimizer: torch.optim.Optimizer, epoch: int, 
               loss: float, filepath: Path) -> None:
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": {
            "latent_dim": config.latent_dim,
            "text_input_dim": None,  # Will be filled by actual dimension
        }
    }, filepath)
    logging.info(f"Model saved to {filepath}")


def main() -> None:
    """Main training function."""
    setup_logging(config.log_level)
    logging.info("Starting encoder training...")
    logging.info(f"Configuration: {config.config_path}")
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    # Get device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load preprocessed data
    train_df = pd.read_csv(config.data_dir / "train.csv")
    val_df = pd.read_csv(config.data_dir / "val.csv")
    text_embeddings = np.load(config.data_dir / "text_embeddings.npy")
    
    logging.info(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples")
    logging.info(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Create datasets
    train_transform = create_data_transforms()
    val_transform = create_validation_transforms()
    
    image_dir = config.data_dir / "images"
    
    train_dataset = RecipeImageDataset(
        train_df, text_embeddings[:len(train_df)], image_dir, train_transform
    )
    val_dataset = RecipeImageDataset(
        val_df, text_embeddings[len(train_df):len(train_df)+len(val_df)], 
        image_dir, val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.encoder_batch_size, 
        shuffle=True, 
        num_workers=config.get("environment.num_workers", 4),
        pin_memory=config.get("environment.pin_memory", True)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.encoder_batch_size, 
        shuffle=False, 
        num_workers=config.get("environment.num_workers", 4),
        pin_memory=config.get("environment.pin_memory", True)
    )
    
    # Initialize model
    text_input_dim = text_embeddings.shape[1]
    model = CLIPMini(text_input_dim, config.latent_dim).to(device)
    
    logging.info(f"Model size: {get_model_size(model)} parameters")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get("models.encoder.learning_rate", 0.001),
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.encoder_epochs
    )
    
    # Training loop
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(config.encoder_epochs):
        logging.info(f"Epoch {epoch + 1}/{config.encoder_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, config.models_dir / "encoder_best.pt")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, epoch, val_loss, config.models_dir / f"encoder_epoch_{epoch+1}.pt")
    
    # Save final model
    save_model(model, optimizer, config.encoder_epochs - 1, val_loss, config.models_dir / "encoder_final.pt")
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    # Print summary
    print("\n" + "="*50)
    print("ENCODER TRAINING COMPLETE")
    print("="*50)
    print(f"🎯 Final validation loss: {best_val_loss:.4f}")
    print(f"⏱️  Training time: {training_time:.1f}s")
    print(f"🧠 Model size: {get_model_size(model)}")
    print(f"📊 Latent dimension: {config.latent_dim}")
    print(f"💾 Model saved to: {config.models_dir}")
    print("="*50)


if __name__ == "__main__":
    main()