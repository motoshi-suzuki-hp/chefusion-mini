#!/usr/bin/env python3
"""
Recipe Export Script

This script exports generated recipes to various formats for analysis.

Usage:
    python export_recipes.py [--format csv|json|html] [--count 100] [--cuisine japanese|italian]
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def export_to_csv(df, output_file, cuisine=None, count=None):
    """Export recipes to CSV format."""
    if cuisine:
        df = df[df['cuisine'] == cuisine]
    
    if count:
        df = df.head(count)
    
    df.to_csv(output_file, index=False)
    print(f"✅ Exported {len(df):,} recipes to {output_file}")


def export_to_json(df, output_file, cuisine=None, count=None):
    """Export recipes to JSON format."""
    if cuisine:
        df = df[df['cuisine'] == cuisine]
    
    if count:
        df = df.head(count)
    
    # Convert to records format
    recipes = []
    for _, row in df.iterrows():
        recipe = {
            "id": int(row.get('id', 0)),
            "title": row['title'],
            "cuisine": row['cuisine'],
            "rating": float(row['rating']),
            "ingredients": [ing.strip() for ing in row['ingredients'].split(',')],
            "instructions": row['instructions']
        }
        
        if 'num_ingredients' in row:
            recipe['num_ingredients'] = int(row['num_ingredients'])
        
        recipes.append(recipe)
    
    export_data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "total_recipes": len(recipes),
            "cuisine_filter": cuisine,
            "count_limit": count
        },
        "recipes": recipes
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(recipes):,} recipes to {output_file}")


def export_to_html(df, output_file, cuisine=None, count=None):
    """Export recipes to HTML format."""
    if cuisine:
        df = df[df['cuisine'] == cuisine]
    
    if count:
        df = df.head(count)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fusion Cuisine Recipes</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            .recipe {{
                border: 1px solid #ddd;
                margin: 20px 0;
                padding: 20px;
                border-radius: 8px;
                background-color: #fafafa;
            }}
            .recipe-title {{
                color: #e74c3c;
                font-size: 1.4em;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .recipe-meta {{
                color: #7f8c8d;
                margin-bottom: 15px;
                font-size: 0.9em;
            }}
            .ingredients {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .instructions {{
                background-color: #fff;
                padding: 15px;
                border-left: 4px solid #3498db;
                margin: 10px 0;
            }}
            .cuisine-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: bold;
                color: white;
            }}
            .japanese {{ background-color: #e74c3c; }}
            .italian {{ background-color: #27ae60; }}
            .rating {{
                color: #f39c12;
                font-weight: bold;
            }}
            .stats {{
                background-color: #3498db;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍽️ Fusion Cuisine Recipe Collection</h1>
            
            <div class="stats">
                <strong>📊 Collection Stats:</strong> 
                {len(df):,} recipes | 
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                {f' | Cuisine: {cuisine.title()}' if cuisine else ''}
            </div>
    """
    
    # Add recipes
    for i, (_, recipe) in enumerate(df.iterrows(), 1):
        cuisine_class = recipe['cuisine'].lower()
        cuisine_emoji = "🍣" if cuisine_class == "japanese" else "🍝"
        
        # Format ingredients as a list
        ingredients_list = [ing.strip() for ing in recipe['ingredients'].split(',')]
        ingredients_html = "<ul>" + "".join(f"<li>{ing}</li>" for ing in ingredients_list) + "</ul>"
        
        html_content += f"""
            <div class="recipe">
                <div class="recipe-title">
                    {cuisine_emoji} {recipe['title']}
                </div>
                <div class="recipe-meta">
                    <span class="cuisine-badge {cuisine_class}">{recipe['cuisine'].title()}</span>
                    <span class="rating">⭐ {recipe['rating']:.1f}/5.0</span>
                    | Recipe #{i}
                </div>
                
                <div class="ingredients">
                    <strong>🛒 Ingredients:</strong>
                    {ingredients_html}
                </div>
                
                <div class="instructions">
                    <strong>👨‍🍳 Instructions:</strong><br>
                    {recipe['instructions']}
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Exported {len(df):,} recipes to {output_file}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export fusion cuisine recipes")
    parser.add_argument("--format", choices=["csv", "json", "html"], default="html",
                        help="Export format (default: html)")
    parser.add_argument("--count", type=int, 
                        help="Number of recipes to export (default: all)")
    parser.add_argument("--cuisine", choices=["japanese", "italian"],
                        help="Filter by cuisine")
    parser.add_argument("--output", type=str,
                        help="Output filename (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # Load recipes
    recipes_path = Path("app/data/recipes.csv")
    
    if not recipes_path.exists():
        print("❌ Recipe file not found!")
        print("Please run data generation first: make data")
        return
    
    df = pd.read_csv(recipes_path)
    print(f"📖 Loaded {len(df):,} recipes")
    
    # Generate output filename if not specified
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cuisine_suffix = f"_{args.cuisine}" if args.cuisine else ""
        count_suffix = f"_{args.count}" if args.count else ""
        output_file = f"recipes{cuisine_suffix}{count_suffix}_{timestamp}.{args.format}"
    
    # Export based on format
    if args.format == "csv":
        export_to_csv(df, output_file, args.cuisine, args.count)
    elif args.format == "json":
        export_to_json(df, output_file, args.cuisine, args.count)
    elif args.format == "html":
        export_to_html(df, output_file, args.cuisine, args.count)
    
    print(f"📁 Output file: {Path(output_file).absolute()}")
    print(f"💡 Open in browser: file://{Path(output_file).absolute()}")


if __name__ == "__main__":
    main()