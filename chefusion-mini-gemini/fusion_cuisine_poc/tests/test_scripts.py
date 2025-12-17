import os
import pytest


def test_imports():
    import scripts.evaluate
    import scripts.generate_fusion
    import scripts.train_encoder
    import scripts.train_palatenet


@pytest.mark.skipif(not os.path.exists("data/recipes.json"), reason="Data not found")
def test_train_encoder():
    import scripts.train_encoder
    scripts.train_encoder.main()


@pytest.mark.skipif(not os.path.exists("data/recipes.json"), reason="Data not found")
def test_train_palatenet():
    import scripts.train_palatenet
    scripts.train_palatenet.main()


@pytest.mark.skipif(not os.path.exists("models/text_encoder.pt"), reason="Model not found")
def test_generate_fusion():
    import scripts.generate_fusion
    scripts.generate_fusion.main()


@pytest.mark.skipif(not os.path.exists("outputs/fused_recipe.txt"), reason="Output not found")
def test_evaluate():
    import scripts.evaluate
    scripts.evaluate.main()
