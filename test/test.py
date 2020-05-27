import numpy as np
import pandas as pd
import logging
import yaml
from os import path
import os

logging.basicConfig(format='%(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('run-reproducibility')

def check_features(features_df, dependent_cols, new_cols):

	"""Ensure that new features and their dependencies are in final features dataframe
	args:
		features_df (obj): 'DataFrame' of final set features
		dependent_cols (obj): 'List' of column names required to generate new features
		new_cols (obj) 'List' of names for new features
	returns:
		feature_check (bool): True if all columns are present in given dataframe
	"""

	# List of features from clean dataframe
	input_cols = features_df.columns.tolist()

	og_check = set(dependent_cols).issubset(set(input_cols))
	new_feat_check = set(dependent_cols).issubset(set(input_cols))

	if og_check & new_feat_check:
		return True
	else:
		return False

home_dir = path.dirname(path.dirname(path.abspath(__file__)))

# Import test configurations from YAML file
config_path = path.join(home_dir, "test/test_config.yaml")

with open(config_path, "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)

# Import features df
features_path = path.join(home_dir, "data/features.csv")
features_df = pd.read_csv(features_path)

# Check features
if check_features(features_df, config['dependent_cols'], config['new_features']):
	logger.info("All features present")
else:
	logger.warning("Not all features present")

# Check model metrics
metrics_path = path.join(home_dir, "metrics/auc_accuracy.txt")
with open(metrics_path, "r") as f:
	metrics = yaml.load(f, Loader=yaml.FullLoader)

if (metrics['AUC'] == config['expected_auc']) & (metrics['Accuracy'] == config['expected_accuracy']):
	logger.info("Model metrics as expected")
else:
	logger.warning("Model not reproduced successfully")


