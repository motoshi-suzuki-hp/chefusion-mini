.PHONY: all data train generate evaluate test clean

all: clean data train generate evaluate

data:
	python scripts/01_fetch_data.py
	python scripts/02_preprocess.py

train:
	python scripts/train_encoder.py
	python scripts/train_palatenet.py

generate:
	python scripts/generate_fusion.py

evaluate:
	python scripts/evaluate.py

test:
	pytest

clean:
	rm -rf data models