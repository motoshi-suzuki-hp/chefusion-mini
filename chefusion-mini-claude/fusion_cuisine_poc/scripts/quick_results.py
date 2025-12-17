#!/usr/bin/env python3
"""
Quick Results Viewer for Fusion Cuisine Project.

This script provides a fast overview of project results without dependencies.
Perfect for quickly checking what has been generated.

Usage:
    python scripts/quick_results.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def check_path(path, description):
    """Check if a path exists and return status info."""
    p = Path(path)
    if p.exists():
        if p.is_file():
            size = p.stat().st_size
            return {
                "status": "✅",
                "description": description,
                "size_mb": size / (1024**2),
                "exists": True
            }
        else:
            files = list(p.glob("*"))
            return {
                "status": "✅",
                "description": f"{description} (directory)",
                "file_count": len(files),
                "exists": True
            }
    else:
        return {
            "status": "❌",
            "description": description,
            "exists": False
        }


def quick_overview():
    """Provide a quick overview of project status."""
    print("🚀 FUSION CUISINE - QUICK RESULTS")
    print("=" * 50)
    print(f"📅 Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Key files to check
    key_files = [
        ("app/data/recipes.csv", "Recipe dataset"),
        ("app/data/data_summary.json", "Data summary"),
        ("app/data/flavor_graph.json", "Flavor graph"),
        ("app/data/text_embeddings.npy", "Text embeddings"),
        ("app/models/encoder_best.pt", "Encoder model"),
        ("app/models/palatenet_best.pt", "PalateNet model"),
        ("outputs/evaluation_metrics.json", "Evaluation results"),
        ("outputs/recipes/fusion_recipes.json", "Fusion recipes"),
        ("logs/fusion_cuisine.log", "Project logs")
    ]
    
    print(f"\n📋 FILE STATUS:")
    print("-" * 35)
    
    total_size = 0
    completed_steps = 0
    
    for filepath, description in key_files:
        status_info = check_path(filepath, description)
        status = status_info["status"]
        desc = status_info["description"]
        
        if status_info["exists"]:
            completed_steps += 1
            if "size_mb" in status_info:
                size_mb = status_info["size_mb"]
                total_size += size_mb
                print(f"{status} {desc}: {size_mb:.1f} MB")
            else:
                count = status_info.get("file_count", 0)
                print(f"{status} {desc}: {count} files")
        else:
            print(f"{status} {desc}: Not found")
    
    # Calculate completion percentage
    completion = (completed_steps / len(key_files)) * 100
    
    print(f"\n📊 PROJECT STATUS:")
    print("-" * 25)
    print(f"Completion: {completion:.0f}% ({completed_steps}/{len(key_files)} files)")
    print(f"Total size: {total_size:.1f} MB")
    
    # Determine project phase
    if completion >= 90:
        phase = "✅ COMPLETE"
        color = "green"
    elif completion >= 60:
        phase = "🔄 PARTIAL"
        color = "yellow" 
    elif completion >= 30:
        phase = "🚧 IN PROGRESS"
        color = "blue"
    else:
        phase = "❌ INCOMPLETE"
        color = "red"
    
    print(f"Phase: {phase}")
    
    return completion, completed_steps, total_size


def show_data_summary():
    """Show data generation summary if available."""
    summary_path = Path("app/data/data_summary.json")
    
    if summary_path.exists():
        print(f"\n📊 DATA SUMMARY:")
        print("-" * 20)
        
        try:
            with open(summary_path) as f:
                data = json.load(f)
            
            print(f"Total recipes: {data.get('total_recipes', 0):,}")
            
            cuisines = data.get('recipes_by_cuisine', {})
            for cuisine, count in cuisines.items():
                emoji = "🍣" if cuisine == "japanese" else "🍝" if cuisine == "italian" else "🍽️"
                print(f"{emoji} {cuisine.capitalize()}: {count:,}")
            
            print(f"Total images: {data.get('total_images', 0):,}")
            print(f"Total ratings: {data.get('total_ratings', 0):,}")
            print(f"Average rating: {data.get('avg_rating', 0):.2f}")
            print(f"Flavor graph: {data.get('flavor_graph_nodes', 0)} nodes")
            
        except Exception as e:
            print(f"Error reading data summary: {e}")


def show_sample_recipes():
    """Show a few sample recipes if available."""
    recipes_path = Path("app/data/recipes.csv")
    
    if recipes_path.exists():
        print(f"\n🍽️ SAMPLE RECIPES:")
        print("-" * 22)
        
        try:
            # Read CSV manually to avoid pandas dependency
            with open(recipes_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > 1:  # Has header + data
                # Parse header
                header = lines[0].strip().split(',')
                
                # Find column indices
                title_idx = header.index('title') if 'title' in header else 0
                cuisine_idx = header.index('cuisine') if 'cuisine' in header else 1
                rating_idx = header.index('rating') if 'rating' in header else 2
                
                # Show first few recipes
                sample_count = min(3, len(lines) - 1)
                for i in range(1, sample_count + 1):
                    parts = lines[i].strip().split(',')
                    if len(parts) > max(title_idx, cuisine_idx, rating_idx):
                        title = parts[title_idx].strip('"')
                        cuisine = parts[cuisine_idx].strip('"')
                        rating = parts[rating_idx].strip('"')
                        
                        emoji = "🍣" if cuisine == "japanese" else "🍝" if cuisine == "italian" else "🍽️"
                        print(f"{emoji} {title} (⭐{rating})")
                
                total_recipes = len(lines) - 1
                print(f"... and {total_recipes - sample_count:,} more recipes")
            
        except Exception as e:
            print(f"Error reading recipes: {e}")


def show_model_status():
    """Show model training status."""
    models_dir = Path("app/models")
    
    if models_dir.exists():
        print(f"\n🧠 MODEL STATUS:")
        print("-" * 18)
        
        model_files = list(models_dir.glob("*.pt"))
        
        if model_files:
            for model_file in sorted(model_files):
                size_mb = model_file.stat().st_size / (1024**2)
                print(f"✅ {model_file.name}: {size_mb:.1f} MB")
        else:
            print("❌ No model files found")


def show_next_steps(completion):
    """Show suggested next steps based on completion status."""
    print(f"\n💡 NEXT STEPS:")
    print("-" * 15)
    
    if completion < 30:
        print("🚀 Run data generation: make data")
        print("🔄 Run preprocessing: make preprocess")
    elif completion < 60:
        print("🧠 Train models: make train")
        print("🎨 Generate fusion recipes: make generate")
    elif completion < 90:
        print("🎨 Complete fusion generation: make generate")
        print("📊 Run evaluation: make evaluate")
    else:
        print("✅ Project complete! Explore results:")
        print("🔍 python scripts/show_results.py")
        print("🍽️ python scripts/recipe_explorer.py --interactive")
        print("📱 Open Jupyter notebook for interactive exploration")


def main():
    """Main function."""
    print("⚡ QUICK RESULTS CHECK")
    print("=" * 30)
    
    # Quick overview
    completion, completed_steps, total_size = quick_overview()
    
    # Detailed sections
    show_data_summary()
    show_sample_recipes() 
    show_model_status()
    
    # Next steps
    show_next_steps(completion)
    
    print(f"\n🔧 USEFUL COMMANDS:")
    print("-" * 20)
    print("📊 Detailed results: python scripts/show_results.py")
    print("🍽️ Browse recipes: python scripts/recipe_explorer.py")
    print("📋 Run pipeline: make all")
    print("🔍 Monitor logs: make monitor-logs")
    print("📱 Jupyter notebook: make jupyter")
    
    print(f"\n✨ Quick check complete!")


if __name__ == "__main__":
    main()