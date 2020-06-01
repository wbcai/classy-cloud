# Cloud classification

## Background
Building a logistic regression model to classify cloud types.
Dataset obtained from UC Irvine's Machine Learning [data repository](https://archive.ics.uci.edu/ml/datasets.php)

## Run app on Docker

Build Docker image and run all programs
```
bash run-docker.sh
```

## Workflow documented with a `Makefile`

Prior to running any functions, make the following directories in the root directory: `data`, `model`, `metrics`

Download dataset from UCI repository
```
make acquire
```
Clean dataset, generate classification labels, and save as a DataFrame
```
make clean
```
Create additional features
```
make featurize
```
Fit model on training dataset and generate test metrics
```
make metrics
```
Fit and save model with full dataset
```
make fit
```
Conduct reproducibility tests
```
make test
```
