#!/usr/bin/env bash

# Acquire data from URL
python3 run.py acquire --config=config.yaml --output=data/cloud.data

# Clean data
python3 run.py clean --config=config.yaml --input=data/cloud.data --output=data/clean.csv

# Create additional features
python3 run.py featurize --config=config.yaml --input=data/clean.csv --output=data/features.csv

# Generate training metrics
python3 run.py test --config=config.yaml --input=data/features.csv --output=metrics/

# Fit model on full dataset
python3 run.py fit --config=config.yaml --input=data/features.csv --output=model/lr.pkl

# Test feature dependencies and model accuracy
python3 test/test.py