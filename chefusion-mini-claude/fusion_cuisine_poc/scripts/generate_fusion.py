#!/usr/bin/env python3
"""Fusion and generation script - Create fusion recipes using trained models."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.utils import (
    get_device,
    load_json,
    save_json,
    set_random_seeds,
    setup_logging,
)

# Import model classes
from train_encoder import CLIPMini


class FusionGenerator:
    """Generator for fusion cuisine recipes and images."""
    
    def __init__(self):
        self.device = get_device()
        self.encoder_model = None
        self.diffusion_model = None
        self.openai_client = None
        
    def load_models(self) -> None:
        """Load trained models."""
        logging.info("Loading trained models...")
        
        # Load encoder model
        encoder_path = config.models_dir / "encoder_best.pt"
        if encoder_path.exists():
            self._load_encoder_model(encoder_path)
        else:
            logging.warning(f"Encoder model not found at {encoder_path}")
        
        # Try to load diffusion model if GPU available
        if config.use_gpu and torch.cuda.is_available():
            self._load_diffusion_model()
        else:
            logging.info("GPU not available or disabled, will use text generation fallback")
        
        # Initialize OpenAI client if API key available
        if config.openai_api_key:
            self._initialize_openai_client()
    
    def _load_encoder_model(self, model_path: Path) -> None:
        """Load the trained encoder model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load text embeddings to get input dimension
        text_embeddings = np.load(config.data_dir / "text_embeddings.npy")
        text_input_dim = text_embeddings.shape[1]
        
        # Initialize model
        self.encoder_model = CLIPMini(text_input_dim, config.latent_dim)
        self.encoder_model.load_state_dict(checkpoint["model_state_dict"])
        self.encoder_model.to(self.device)
        self.encoder_model.eval()
        
        logging.info(f"Loaded encoder model from {model_path}")
    
    def _load_diffusion_model(self) -> None:
        """Load Stable Diffusion XL model."""
        try:
            from diffusers import StableDiffusionXLPipeline
            
            model_name = config.get("models.generation.stable_diffusion_model", "stabilityai/stable-diffusion-xl-base-1.0")
            
            self.diffusion_model = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            )
            
            if torch.cuda.is_available():
                self.diffusion_model = self.diffusion_model.to("cuda")
            
            logging.info(f"Loaded Stable Diffusion XL model: {model_name}")
            
        except Exception as e:
            logging.warning(f"Failed to load Stable Diffusion model: {e}")
            self.diffusion_model = None
    
    def _initialize_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self.openai_client = OpenAI(api_key=config.openai_api_key)
            logging.info("Initialized OpenAI client")
            
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
    
    def extract_cuisine_embeddings(self) -> Dict[str, torch.Tensor]:
        """Extract representative embeddings for each cuisine."""
        logging.info("Extracting cuisine embeddings...")
        
        # Load preprocessed data
        train_df = pd.read_csv(config.data_dir / "train.csv")
        text_embeddings = np.load(config.data_dir / "text_embeddings.npy")
        
        cuisine_embeddings = {}
        
        for cuisine in config.target_cuisines:
            cuisine_recipes = train_df[train_df["cuisine"] == cuisine]
            cuisine_indices = cuisine_recipes.index.tolist()
            
            if self.encoder_model is not None:
                # Use trained encoder to get embeddings
                cuisine_text_embeddings = text_embeddings[cuisine_indices]
                cuisine_text_tensor = torch.tensor(cuisine_text_embeddings, dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    # Get text embeddings from the encoder
                    encoded_embeddings = self.encoder_model.text_encoder(cuisine_text_tensor)
                
                # Average the embeddings for this cuisine
                avg_embedding = encoded_embeddings.mean(dim=0)
            else:
                # Fallback: use original text embeddings
                cuisine_text_embeddings = text_embeddings[cuisine_indices]
                avg_embedding = torch.tensor(cuisine_text_embeddings.mean(axis=0), dtype=torch.float32)
            
            cuisine_embeddings[cuisine] = avg_embedding
            logging.info(f"Extracted {cuisine} embedding: {avg_embedding.shape}")
        
        return cuisine_embeddings
    
    def create_fusion_embeddings(self, cuisine_embeddings: Dict[str, torch.Tensor]) -> Dict[float, torch.Tensor]:
        """Create fusion embeddings with different alpha values."""
        logging.info("Creating fusion embeddings...")
        
        jp_embedding = cuisine_embeddings["japanese"]
        it_embedding = cuisine_embeddings["italian"]
        
        fusion_embeddings = {}
        
        for alpha in config.fusion_alphas:
            # v_fusion = α * v_JP + (1-α) * v_IT
            fusion_embedding = alpha * jp_embedding + (1 - alpha) * it_embedding
            fusion_embedding = F.normalize(fusion_embedding, p=2, dim=0)
            
            fusion_embeddings[alpha] = fusion_embedding
            logging.info(f"Created fusion embedding for α={alpha}")
        
        return fusion_embeddings
    
    def generate_fusion_descriptions(self, fusion_embeddings: Dict[float, torch.Tensor]) -> Dict[float, str]:
        """Generate textual descriptions for fusion embeddings."""
        descriptions = {}
        
        for alpha, embedding in fusion_embeddings.items():
            if alpha < 0.4:
                style = "predominantly Italian with subtle Japanese influences"
                techniques = "Italian cooking methods enhanced with Japanese umami and presentation"
            elif alpha > 0.6:
                style = "predominantly Japanese with Italian ingredients and flavors"
                techniques = "Japanese cooking techniques using Italian ingredients"
            else:
                style = "balanced fusion of Japanese and Italian cuisines"
                techniques = "harmonious blend of Japanese and Italian cooking methods"
            
            description = f"A {style}, featuring {techniques}. "
            description += f"This fusion combines the best of both culinary traditions."
            
            descriptions[alpha] = description
        
        return descriptions
    
    def generate_recipe_prompt(self, alpha: float, fusion_description: str) -> str:
        """Generate a prompt for recipe generation."""
        return f"""Create a detailed fusion recipe that combines Japanese and Italian cuisines with a {alpha:.1f} Japanese to {1-alpha:.1f} Italian ratio.

Style: {fusion_description}

Please provide:
1. Recipe title
2. Ingredients list (with quantities)
3. Step-by-step cooking instructions
4. Cooking time and servings
5. Brief description of the fusion concept

The recipe should be authentic, practical, and showcase the unique combination of flavors and techniques from both cuisines."""
    
    def generate_image_prompt(self, alpha: float, fusion_description: str) -> str:
        """Generate a prompt for image generation."""
        if alpha < 0.4:
            visual_style = "Italian plating with Japanese garnish and presentation"
        elif alpha > 0.6:
            visual_style = "Japanese plating with Italian ingredients and colors"
        else:
            visual_style = "elegant fusion plating combining Japanese minimalism with Italian abundance"
        
        return f"""A beautiful, professional food photograph of a fusion dish combining Japanese and Italian cuisines. {fusion_description}. The presentation features {visual_style}. High-quality, restaurant-style plating, natural lighting, appetizing, detailed, 4K resolution."""
    
    def generate_text_recipes(self, fusion_embeddings: Dict[float, torch.Tensor]) -> Dict[float, str]:
        """Generate text recipes using OpenAI API."""
        logging.info("Generating text recipes...")
        
        if not self.openai_client:
            logging.warning("OpenAI client not available, generating simple recipes")
            return self._generate_simple_recipes(fusion_embeddings)
        
        fusion_descriptions = self.generate_fusion_descriptions(fusion_embeddings)
        recipes = {}
        
        for alpha in tqdm(config.fusion_alphas, desc="Generating recipes"):
            try:
                prompt = self.generate_recipe_prompt(alpha, fusion_descriptions[alpha])
                
                response = self.openai_client.chat.completions.create(
                    model=config.get("models.generation.openai_model", "gpt-4-turbo"),
                    messages=[
                        {"role": "system", "content": "You are a professional chef specializing in fusion cuisine."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=config.get("models.generation.max_tokens", 1000),
                    temperature=config.get("models.generation.temperature", 0.7)
                )
                
                recipe_text = response.choices[0].message.content
                recipes[alpha] = recipe_text
                
                logging.info(f"Generated recipe for α={alpha}")
                
            except Exception as e:
                logging.error(f"Failed to generate recipe for α={alpha}: {e}")
                recipes[alpha] = self._generate_simple_recipe(alpha, fusion_descriptions[alpha])
        
        return recipes
    
    def _generate_simple_recipes(self, fusion_embeddings: Dict[float, torch.Tensor]) -> Dict[float, str]:
        """Generate simple fallback recipes."""
        recipes = {}
        
        for alpha in config.fusion_alphas:
            if alpha < 0.4:
                recipe = f"""
Italian-Japanese Fusion Pasta (α={alpha})

Ingredients:
- 300g spaghetti
- 200g salmon fillet
- 2 tbsp soy sauce
- 1 tbsp miso paste
- 200ml heavy cream
- 2 sheets nori, sliced
- 2 green onions, chopped
- 1 tbsp olive oil

Instructions:
1. Cook spaghetti according to package directions
2. Pan-sear salmon and flake into pieces
3. Mix soy sauce, miso, and cream
4. Toss pasta with sauce and salmon
5. Garnish with nori and green onions

Serves 4 | Prep: 15 min | Cook: 20 min
"""
            elif alpha > 0.6:
                recipe = f"""
Japanese-Italian Fusion Rice Bowl (α={alpha})

Ingredients:
- 2 cups sushi rice
- 200g pancetta, diced
- 1 cup cherry tomatoes
- 100g parmesan cheese
- 2 tbsp mirin
- 1 tbsp rice vinegar
- Fresh basil leaves
- Sesame seeds

Instructions:
1. Prepare sushi rice with mirin and vinegar
2. Cook pancetta until crispy
3. Sauté cherry tomatoes
4. Serve rice topped with pancetta and tomatoes
5. Garnish with parmesan, basil, and sesame seeds

Serves 4 | Prep: 20 min | Cook: 25 min
"""
            else:
                recipe = f"""
Balanced Fusion Ramen (α={alpha})

Ingredients:
- 4 portions fresh ramen noodles
- 1L chicken and kombu broth
- 200g Italian sausage
- 2 soft-boiled eggs
- 100g fresh mozzarella
- 2 tbsp white miso
- Cherry tomatoes
- Fresh basil and nori

Instructions:
1. Prepare rich broth with miso
2. Cook and slice Italian sausage
3. Boil ramen noodles
4. Assemble bowls with noodles, broth, sausage
5. Top with mozzarella, egg, tomatoes, herbs

Serves 4 | Prep: 30 min | Cook: 45 min
"""
            
            recipes[alpha] = recipe.strip()
        
        return recipes
    
    def _generate_simple_recipe(self, alpha: float, description: str) -> str:
        """Generate a single simple recipe."""
        return self._generate_simple_recipes({alpha: None})[alpha]
    
    def generate_images(self, fusion_embeddings: Dict[float, torch.Tensor]) -> Dict[float, Optional[Image.Image]]:
        """Generate images using Stable Diffusion XL."""
        logging.info("Generating fusion recipe images...")
        
        if not self.diffusion_model:
            logging.warning("Diffusion model not available, skipping image generation")
            return {alpha: None for alpha in config.fusion_alphas}
        
        fusion_descriptions = self.generate_fusion_descriptions(fusion_embeddings)
        images = {}
        
        for alpha in tqdm(config.fusion_alphas, desc="Generating images"):
            try:
                prompt = self.generate_image_prompt(alpha, fusion_descriptions[alpha])
                
                # Generate image
                result = self.diffusion_model(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                )
                
                images[alpha] = result.images[0]
                logging.info(f"Generated image for α={alpha}")
                
            except Exception as e:
                logging.error(f"Failed to generate image for α={alpha}: {e}")
                images[alpha] = None
        
        return images
    
    def save_outputs(self, recipes: Dict[float, str], images: Dict[float, Optional[Image.Image]],
                    fusion_embeddings: Dict[float, torch.Tensor]) -> None:
        """Save all generated outputs."""
        logging.info("Saving fusion outputs...")
        
        # Save recipes
        recipes_output = {}
        for alpha, recipe in recipes.items():
            filename = f"fusion_recipe_alpha_{alpha:.1f}.txt"
            filepath = config.outputs_dir / "recipes" / filename
            
            with open(filepath, "w") as f:
                f.write(recipe)
            
            recipes_output[f"alpha_{alpha:.1f}"] = {
                "alpha": alpha,
                "recipe": recipe,
                "file": str(filepath)
            }
        
        # Save recipe metadata
        save_json(recipes_output, config.outputs_dir / "recipes" / "fusion_recipes.json")
        
        # Save images
        images_output = {}
        for alpha, image in images.items():
            if image is not None:
                filename = f"fusion_image_alpha_{alpha:.1f}.png"
                filepath = config.outputs_dir / "images" / filename
                image.save(filepath)
                
                images_output[f"alpha_{alpha:.1f}"] = {
                    "alpha": alpha,
                    "file": str(filepath),
                    "size": image.size
                }
        
        # Save image metadata
        if images_output:
            save_json(images_output, config.outputs_dir / "images" / "fusion_images.json")
        
        # Save embeddings
        embeddings_output = {}
        for alpha, embedding in fusion_embeddings.items():
            embeddings_output[f"alpha_{alpha:.1f}"] = {
                "alpha": alpha,
                "embedding": embedding.cpu().numpy().tolist(),
                "dimension": embedding.size(0)
            }
        
        save_json(embeddings_output, config.outputs_dir / "fusion_embeddings.json")
        
        logging.info("All outputs saved successfully!")


def main() -> None:
    """Main fusion generation function."""
    setup_logging(config.log_level)
    logging.info("Starting fusion generation...")
    logging.info(f"Configuration: {config.config_path}")
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    # Initialize generator
    generator = FusionGenerator()
    
    # Load models
    generator.load_models()
    
    # Extract cuisine embeddings
    cuisine_embeddings = generator.extract_cuisine_embeddings()
    
    # Create fusion embeddings
    fusion_embeddings = generator.create_fusion_embeddings(cuisine_embeddings)
    
    # Generate text recipes
    start_time = time.time()
    recipes = generator.generate_text_recipes(fusion_embeddings)
    recipe_time = time.time() - start_time
    
    # Generate images (if possible)
    start_time = time.time()
    images = generator.generate_images(fusion_embeddings)
    image_time = time.time() - start_time
    
    # Save outputs
    generator.save_outputs(recipes, images, fusion_embeddings)
    
    # Print summary
    print("\n" + "="*50)
    print("FUSION GENERATION COMPLETE")
    print("="*50)
    print(f"🎯 Alpha values: {config.fusion_alphas}")
    print(f"📝 Recipes generated: {len([r for r in recipes.values() if r])}")
    print(f"🖼️  Images generated: {len([i for i in images.values() if i is not None])}")
    print(f"⏱️  Recipe generation time: {recipe_time:.1f}s")
    print(f"⏱️  Image generation time: {image_time:.1f}s")
    print(f"💾 Outputs saved to: {config.outputs_dir}")
    print(f"🔧 GPU used: {torch.cuda.is_available() and config.use_gpu}")
    print(f"🔑 OpenAI API used: {config.openai_api_key is not None}")
    print("="*50)


if __name__ == "__main__":
    main()