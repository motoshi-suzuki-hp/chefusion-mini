# Multicultural Fusion Cuisine Generator & Palate Predictor

This project is a minimal, end-to-end prototype of a pipeline for generating multicultural fusion cuisine recipes and predicting their taste.

## Quick Start

1.  **Build the Docker image:**

    ```bash
    docker compose build
    ```

2.  **Run the entire pipeline:**

    ```bash
    docker compose run app bash -c "make all"
    ```

    This command will:

    *   Download and preprocess the data.
    *   Train the text encoder and PalateNet models.
    *   Generate a fused recipe.
    *   Evaluate the models and print the metrics.

## Hardware/OS Assumptions

*   This project is designed to be run in a Docker container, so it should be compatible with any OS that supports Docker.
*   A GPU is recommended for training the models, but the code will fall back to CPU if a GPU is not available.

## Expected Runtime & Metrics

*   **Expected Runtime:** The entire pipeline should run in under 4 hours on a modern laptop.
*   **Expected Metrics:**
    *   **PalateNet Spearman Rho:** > 0.3
    *   **Ingredient Overlap:** > 0.1 for both Japanese and Italian cuisines.

### Sample Output

```json
{
    "palatenet_spearman_rho": 0.35,
    "generated_recipe_jp_ingredient_overlap": 0.15,
    "generated_recipe_it_ingredient_overlap": 0.20
}
```
