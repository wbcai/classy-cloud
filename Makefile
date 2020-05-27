acquire: config.yaml
	python3 run.py acquire --config=config.yaml --output=data/cloud.data

clean: acquire config.yaml
	python3 run.py clean --config=config.yaml --input=data/cloud.data --output=data/clean.csv

featurize: clean config.yaml
	python3 run.py featurize --config=config.yaml --input=data/clean.csv --output=data/features.csv

metrics: featurize config.yaml
	python3 run.py test --config=config.yaml --input=data/features.csv --output=metrics/

fit: featurize config.yaml
	python3 run.py fit --config=config.yaml --input=data/features.csv --output=model/lr.pkl

test: metrics
	python3 test/test.py

all: acquire clean featurize metrics fit test

clear:
	rm data/*
	rm metrics/*
	rm model/*

.PHONY: clear all