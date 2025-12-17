#!/usr/bin/env python3
"""
Results display script for the Fusion Cuisine project.

This script provides a comprehensive view of all project results,
including data generation, model training, and analysis.

Usage:
    python scripts/show_results.py [--format json|table|html] [--export]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import setup_logging, format_time


def load_config():
    """Load results configuration."""
    config_path = Path("results_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def check_file_status(file_path):
    """Check if a file exists and return its status."""
    path = Path(file_path)
    if path.exists():
        if path.is_file():
            size = path.stat().st_size
            return {
                "exists": True,
                "size_bytes": size,
                "size_mb": size / (1024**2),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
        else:
            files = list(path.glob("*"))
            return {
                "exists": True,
                "is_directory": True,
                "file_count": len(files),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
    return {"exists": False}


def analyze_data_generation(config):
    """Analyze data generation results."""
    results = {"status": "unknown", "data": {}}
    
    summary_path = config.get("data_paths", {}).get("data_summary")
    if summary_path and Path(summary_path).exists():
        with open(summary_path) as f:
            data_summary = json.load(f)
        
        results["status"] = "complete"
        results["data"] = data_summary
        
        # Add file status for key data files
        data_files = config.get("data_paths", {})
        results["file_status"] = {}
        for name, path in data_files.items():
            results["file_status"][name] = check_file_status(path)
    
    return results


def analyze_model_training(config):
    """Analyze model training results."""
    results = {"encoder": {}, "palatenet": {}}
    
    model_files = config.get("model_paths", {})
    
    for model_type in ["encoder", "palatenet"]:
        best_path = model_files.get(f"{model_type}_best")
        final_path = model_files.get(f"{model_type}_final")
        
        results[model_type] = {
            "best_model": check_file_status(best_path) if best_path else {"exists": False},
            "final_model": check_file_status(final_path) if final_path else {"exists": False}
        }
        
        # Determine training status
        if results[model_type]["best_model"]["exists"]:
            results[model_type]["status"] = "complete"
        else:
            results[model_type]["status"] = "not_trained"
    
    return results


def analyze_recipe_data(config):
    """Analyze recipe data and show samples."""
    results = {"samples": {}, "statistics": {}}
    
    recipes_path = config.get("data_paths", {}).get("recipes")
    if recipes_path and Path(recipes_path).exists():
        df = pd.read_csv(recipes_path)
        
        # Basic statistics
        results["statistics"] = {
            "total_recipes": len(df),
            "cuisines": df["cuisine"].unique().tolist(),
            "recipes_by_cuisine": df["cuisine"].value_counts().to_dict(),
            "average_rating": float(df["rating"].mean()),
            "rating_range": [float(df["rating"].min()), float(df["rating"].max())],
            "avg_ingredients": float(df["num_ingredients"].mean()) if "num_ingredients" in df.columns else None
        }
        
        # Sample recipes
        sample_count = config.get("analysis", {}).get("sample_recipes_per_cuisine", 3)
        for cuisine in df["cuisine"].unique():
            cuisine_recipes = df[df["cuisine"] == cuisine].head(sample_count)
            results["samples"][cuisine] = []
            
            for _, recipe in cuisine_recipes.iterrows():
                sample = {
                    "title": recipe["title"],
                    "rating": float(recipe["rating"]),
                    "ingredients": recipe["ingredients"][:200] + "..." if len(recipe["ingredients"]) > 200 else recipe["ingredients"],
                    "instructions": recipe["instructions"][:300] + "..." if len(recipe["instructions"]) > 300 else recipe["instructions"]
                }
                results["samples"][cuisine].append(sample)
    
    return results


def analyze_embeddings(config):
    """Analyze embedding data."""
    results = {}
    
    text_emb_path = config.get("data_paths", {}).get("text_embeddings")
    if text_emb_path and Path(text_emb_path).exists():
        embeddings = np.load(text_emb_path)
        results["text_embeddings"] = {
            "shape": embeddings.shape,
            "num_recipes": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
            "file_size_mb": Path(text_emb_path).stat().st_size / (1024**2)
        }
    
    ing_emb_path = config.get("data_paths", {}).get("ingredient_embeddings")
    if ing_emb_path and Path(ing_emb_path).exists():
        embeddings = np.load(ing_emb_path)
        results["ingredient_embeddings"] = {
            "shape": embeddings.shape,
            "num_ingredients": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
            "file_size_mb": Path(ing_emb_path).stat().st_size / (1024**2)
        }
    
    return results


def generate_results_summary(config):
    """Generate comprehensive results summary."""
    summary = {
        "project_info": config.get("project", {}),
        "timestamp": datetime.now().isoformat(),
        "data_generation": analyze_data_generation(config),
        "model_training": analyze_model_training(config),
        "recipe_analysis": analyze_recipe_data(config),
        "embeddings": analyze_embeddings(config)
    }
    
    # Overall status
    data_complete = summary["data_generation"]["status"] == "complete"
    encoder_complete = summary["model_training"]["encoder"]["status"] == "complete"
    palatenet_complete = summary["model_training"]["palatenet"]["status"] == "complete"
    
    if data_complete and encoder_complete and palatenet_complete:
        overall_status = "complete"
    elif data_complete and encoder_complete:
        overall_status = "partial_complete"
    elif data_complete:
        overall_status = "data_only"
    else:
        overall_status = "incomplete"
    
    summary["overall_status"] = overall_status
    
    return summary


def print_table_format(summary):
    """Print results in table format."""
    print("🎉 FUSION CUISINE PROJECT RESULTS")
    print("=" * 60)
    
    # Project info
    project = summary.get("project_info", {})
    print(f"\n📋 PROJECT: {project.get('name', 'Unknown')}")
    print(f"   Version: {project.get('version', 'Unknown')}")
    print(f"   Status: {summary['overall_status'].upper()}")
    print(f"   Generated: {summary['timestamp'][:19]}")
    
    # Data generation
    data_gen = summary.get("data_generation", {})
    print(f"\n📊 DATA GENERATION: {data_gen.get('status', 'unknown').upper()}")
    if data_gen.get("data"):
        data = data_gen["data"]
        print(f"   Recipes: {data.get('total_recipes', 0):,}")
        print(f"   Images: {data.get('total_images', 0):,}")
        print(f"   Ratings: {data.get('total_ratings', 0):,}")
        print(f"   Flavor graph: {data.get('flavor_graph_nodes', 0)} nodes")
    
    # Model training
    models = summary.get("model_training", {})
    print(f"\n🧠 MODEL TRAINING:")
    for model_type, model_info in models.items():
        status = model_info.get("status", "unknown")
        print(f"   {model_type.capitalize()}: {status.upper()}")
        if model_info.get("best_model", {}).get("exists"):
            size_mb = model_info["best_model"].get("size_mb", 0)
            print(f"      Size: {size_mb:.1f} MB")
    
    # Recipe analysis
    recipes = summary.get("recipe_analysis", {})
    if recipes.get("statistics"):
        stats = recipes["statistics"]
        print(f"\n🍽️ RECIPE ANALYSIS:")
        print(f"   Total recipes: {stats.get('total_recipes', 0):,}")
        print(f"   Cuisines: {', '.join(stats.get('cuisines', []))}")
        print(f"   Average rating: {stats.get('average_rating', 0):.2f}")
        
        # Show sample recipes
        if recipes.get("samples"):
            print(f"\n📝 SAMPLE RECIPES:")
            for cuisine, samples in recipes["samples"].items():
                print(f"\n   🎌 {cuisine.upper()}:")
                for i, sample in enumerate(samples[:2], 1):  # Show first 2
                    print(f"      {i}. {sample['title']} (⭐{sample['rating']:.1f})")
    
    # Embeddings
    embeddings = summary.get("embeddings", {})
    if embeddings:
        print(f"\n🔢 EMBEDDINGS:")
        if "text_embeddings" in embeddings:
            text_emb = embeddings["text_embeddings"]
            print(f"   Text: {text_emb['num_recipes']:,} recipes × {text_emb['embedding_dim']} dims")
        if "ingredient_embeddings" in embeddings:
            ing_emb = embeddings["ingredient_embeddings"]
            print(f"   Ingredients: {ing_emb['num_ingredients']} × {ing_emb['embedding_dim']} dims")


def export_results(summary, format_type="json"):
    """Export results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == "json":
        output_file = f"results_summary_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    elif format_type == "html":
        output_file = f"results_summary_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fusion Cuisine Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E8B57; }}
                h2 {{ color: #4682B4; }}
                .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .complete {{ background-color: #d4edda; }}
                .partial {{ background-color: #fff3cd; }}
                .incomplete {{ background-color: #f8d7da; }}
                .recipe {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>🍜🍝 Fusion Cuisine Project Results</h1>
            <p><strong>Generated:</strong> {summary['timestamp']}</p>
            <p><strong>Status:</strong> <span class="status {summary['overall_status']}">{summary['overall_status'].upper()}</span></p>
        """
        
        # Add sections based on available data
        project = summary.get("project_info", {})
        if project:
            html_content += f"""
            <h2>📋 Project Information</h2>
            <ul>
                <li><strong>Name:</strong> {project.get('name', 'Unknown')}</li>
                <li><strong>Version:</strong> {project.get('version', 'Unknown')}</li>
                <li><strong>Last Run:</strong> {project.get('last_run', 'Unknown')}</li>
            </ul>
            """
        
        # Add data generation results
        data_gen = summary.get("data_generation", {})
        if data_gen.get("data"):
            data = data_gen["data"]
            html_content += f"""
            <h2>📊 Data Generation Results</h2>
            <ul>
                <li><strong>Total Recipes:</strong> {data.get('total_recipes', 0):,}</li>
                <li><strong>Total Images:</strong> {data.get('total_images', 0):,}</li>
                <li><strong>Total Ratings:</strong> {data.get('total_ratings', 0):,}</li>
                <li><strong>Average Rating:</strong> {data.get('avg_rating', 0):.2f}</li>
                <li><strong>Flavor Graph:</strong> {data.get('flavor_graph_nodes', 0)} nodes, {data.get('flavor_graph_edges', 0)} edges</li>
            </ul>
            """
        
        # Add recipe samples
        recipes = summary.get("recipe_analysis", {})
        if recipes.get("samples"):
            html_content += "<h2>🍽️ Sample Recipes</h2>"
            for cuisine, samples in recipes["samples"].items():
                html_content += f"<h3>🎌 {cuisine.upper()}</h3>"
                for sample in samples[:3]:  # Show first 3
                    html_content += f"""
                    <div class="recipe">
                        <h4>{sample['title']} (⭐{sample['rating']:.1f})</h4>
                        <p><strong>Ingredients:</strong> {sample['ingredients']}</p>
                        <p><strong>Instructions:</strong> {sample['instructions']}</p>
                    </div>
                    """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, "w") as f:
            f.write(html_content)
    
    print(f"✅ Results exported to: {output_file}")
    return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Display Fusion Cuisine project results")
    parser.add_argument("--format", choices=["json", "table", "html"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--export", action="store_true",
                        help="Export results to file")
    parser.add_argument("--config", default="results_config.yaml",
                        help="Results configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    # Load configuration
    config = load_config()
    
    # Generate results summary
    summary = generate_results_summary(config)
    
    # Display results
    if args.format == "table":
        print_table_format(summary)
    elif args.format == "json":
        print(json.dumps(summary, indent=2, default=str))
    
    # Export if requested
    if args.export:
        export_results(summary, args.format)
    
    # Show next steps
    print(f"\n💡 NEXT STEPS:")
    print("=" * 30)
    
    status = summary["overall_status"]
    if status == "complete":
        print("✅ Project fully complete! Explore the results.")
        print("🔍 Use: python scripts/show_results.py --export")
        print("📊 Open the Jupyter notebook for interactive exploration")
    elif status == "partial_complete":
        print("⚠️ Project partially complete. Consider:")
        print("🔧 Fix the PalateNet training issue")
        print("🚀 Run fusion recipe generation")
        print("📈 Complete the evaluation phase")
    else:
        print("❌ Project incomplete. Run the pipeline:")
        print("🚀 Use: make all")
        print("📱 Or use the interactive Jupyter notebook")


if __name__ == "__main__":
    main()