#!/usr/bin/env python3
"""
Fix script for image processing warnings in fusion cuisine training.

This script addresses the "Python integer X out of bounds for uint8" warnings
by regenerating corrupted images and validating image data integrity.

Usage:
    python scripts/fix_image_warnings.py
    python scripts/fix_image_warnings.py --validate-only
    python scripts/fix_image_warnings.py --regenerate-all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import random

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config
from app.utils import setup_logging


def create_safe_dummy_image(size: int, base_color: Tuple[int, int, int]) -> Image.Image:
    """Create a safe dummy recipe image with validated color values."""
    # Ensure color values are strictly within valid range [0, 255]
    base_color = tuple(max(0, min(255, int(c))) for c in base_color)
    
    # Create image with base color
    image = Image.new("RGB", (size, size), base_color)
    
    # Add some simple patterns to make images distinguishable
    draw = ImageDraw.Draw(image)
    
    # Add safe color variations (±20 from base color to stay within bounds)
    for _ in range(random.randint(2, 5)):
        # Create safe color variations
        color_var = tuple(
            max(0, min(255, int(base_color[i] + random.randint(-20, 20)))) 
            for i in range(3)
        )
        
        # Add simple shapes
        if random.choice([True, False]):
            # Circle
            center_x, center_y = random.randint(size//4, 3*size//4), random.randint(size//4, 3*size//4)
            radius = random.randint(size//10, size//6)
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], fill=color_var)
        else:
            # Rectangle
            x1, y1 = random.randint(0, size//2), random.randint(0, size//2)
            width, height = random.randint(size//8, size//4), random.randint(size//8, size//4)
            x2, y2 = min(x1 + width, size), min(y1 + height, size)
            draw.rectangle([x1, y1, x2, y2], fill=color_var)
    
    return image


def validate_image(image_path: Path) -> bool:
    """Validate that an image can be loaded without errors."""
    try:
        with Image.open(image_path) as img:
            # Try to convert to RGB and get pixel data
            img_rgb = img.convert("RGB")
            # Force loading the image data
            img_rgb.load()
            # Check basic properties
            if img_rgb.size[0] <= 0 or img_rgb.size[1] <= 0:
                return False
            return True
    except Exception as e:
        logging.debug(f"Image validation failed for {image_path}: {e}")
        return False


def find_corrupted_images(images_dir: Path) -> List[Path]:
    """Find all corrupted or invalid images."""
    corrupted_images = []
    
    if not images_dir.exists():
        logging.warning(f"Images directory does not exist: {images_dir}")
        return corrupted_images
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    
    logging.info(f"Validating {len(image_files)} images...")
    
    for image_path in tqdm(image_files, desc="Validating images"):
        if not validate_image(image_path):
            corrupted_images.append(image_path)
    
    return corrupted_images


def regenerate_corrupted_images(corrupted_images: List[Path], recipes_df: pd.DataFrame) -> int:
    """Regenerate corrupted images with safe dummy images."""
    if not corrupted_images:
        logging.info("No corrupted images found to regenerate")
        return 0
    
    # Define safe color palettes for different cuisines
    color_palettes = {
        "japanese": [(139, 69, 19), (160, 82, 45), (210, 180, 140), (205, 133, 63)],
        "italian": [(178, 34, 34), (220, 20, 60), (255, 140, 0), (255, 99, 71)]
    }
    
    regenerated_count = 0
    
    for image_path in tqdm(corrupted_images, desc="Regenerating images"):
        try:
            # Extract recipe info from filename
            image_filename = image_path.stem
            recipe = recipes_df[recipes_df["image_path"] == f"{image_filename}.jpg"]
            
            if not recipe.empty:
                cuisine = recipe.iloc[0]["cuisine"]
                color = random.choice(color_palettes.get(cuisine, [(128, 128, 128)]))
            else:
                # Default color if recipe not found
                color = (128, 128, 128)
            
            # Create safe dummy image
            image = create_safe_dummy_image(config.image_size, color)
            
            # Save with high quality
            image.save(image_path, "JPEG", quality=95)
            regenerated_count += 1
            
        except Exception as e:
            logging.error(f"Failed to regenerate image {image_path}: {e}")
    
    return regenerated_count


def regenerate_all_images(recipes_df: pd.DataFrame) -> int:
    """Regenerate all images with safe dummy images."""
    images_dir = config.data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Define safe color palettes
    color_palettes = {
        "japanese": [(139, 69, 19), (160, 82, 45), (210, 180, 140), (205, 133, 63)],
        "italian": [(178, 34, 34), (220, 20, 60), (255, 140, 0), (255, 99, 71)]
    }
    
    regenerated_count = 0
    
    for _, recipe in tqdm(recipes_df.iterrows(), total=len(recipes_df), desc="Regenerating all images"):
        image_path = images_dir / recipe["image_path"]
        
        try:
            cuisine = recipe["cuisine"]
            color = random.choice(color_palettes.get(cuisine, [(128, 128, 128)]))
            
            # Create safe dummy image
            image = create_safe_dummy_image(config.image_size, color)
            
            # Save with high quality
            image.save(image_path, "JPEG", quality=95)
            regenerated_count += 1
            
        except Exception as e:
            logging.error(f"Failed to regenerate image {image_path}: {e}")
    
    return regenerated_count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fix image processing warnings in fusion cuisine training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fix_image_warnings.py                    # Fix corrupted images only
  python scripts/fix_image_warnings.py --validate-only    # Just check for issues
  python scripts/fix_image_warnings.py --regenerate-all   # Regenerate all images
        """
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate images, do not regenerate'
    )
    
    parser.add_argument(
        '--regenerate-all',
        action='store_true',
        help='Regenerate all images (not just corrupted ones)'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load recipes data
    recipes_path = config.data_dir / "recipes.csv"
    if not recipes_path.exists():
        logging.error(f"Recipes file not found: {recipes_path}")
        logging.error("Please run data generation first: python scripts/01_fetch_data.py")
        sys.exit(1)
    
    recipes_df = pd.read_csv(recipes_path)
    logging.info(f"Loaded {len(recipes_df)} recipes")
    
    images_dir = config.data_dir / "images"
    
    if args.regenerate_all:
        # Regenerate all images
        logging.info("Regenerating all images with safe dummy images...")
        regenerated_count = regenerate_all_images(recipes_df)
        logging.info(f"Successfully regenerated {regenerated_count} images")
        
    else:
        # Find and fix corrupted images
        corrupted_images = find_corrupted_images(images_dir)
        
        if corrupted_images:
            logging.warning(f"Found {len(corrupted_images)} corrupted images")
            for img_path in corrupted_images[:10]:  # Show first 10
                logging.warning(f"  - {img_path.name}")
            if len(corrupted_images) > 10:
                logging.warning(f"  ... and {len(corrupted_images) - 10} more")
        else:
            logging.info("No corrupted images found!")
        
        if not args.validate_only and corrupted_images:
            # Regenerate corrupted images
            regenerated_count = regenerate_corrupted_images(corrupted_images, recipes_df)
            logging.info(f"Successfully regenerated {regenerated_count} corrupted images")
        
        elif args.validate_only:
            logging.info("Validation complete. Use without --validate-only to fix issues.")
    
    # Final validation
    logging.info("Running final validation...")
    final_corrupted = find_corrupted_images(images_dir)
    
    if final_corrupted:
        logging.error(f"Still {len(final_corrupted)} corrupted images after fix attempt")
        sys.exit(1)
    else:
        logging.info("✅ All images are now valid!")
        print("\n" + "="*50)
        print("IMAGE VALIDATION COMPLETE")
        print("="*50)
        print("✅ All images are valid and safe for training")
        print("✅ No more 'out of bounds' warnings expected")
        print("="*50)


if __name__ == '__main__':
    main()