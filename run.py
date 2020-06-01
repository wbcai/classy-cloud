import argparse
import logging
import yaml
import pandas as pd

from sklearn import model_selection

from src.data_prep import *
from src.modeling import *


logging.basicConfig(format='%(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('run-reproducibility')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Develop model to predict cloud types")

	parser.add_argument('step', help='Which step to run', choices=['acquire', 'clean', 'featurize', 'test', 'fit'])

	parser.add_argument('--input', '-i', default=None, help='Path to input')
	parser.add_argument('--config', '-c', help='Path to configuration file')
	parser.add_argument('--output', '-o', default=None, help='Path to save output; default = None)')

	args = parser.parse_args()

	# Import configurations from YAML file
	with open(args.config, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# Paths based on step arguement	
	if args.step == 'acquire':
		url = config['url']
		path = args.output
		get_raw_data(url, path)

	elif args.step == 'clean':
		clean_data(args.input, args.output, **config['clean'])

	elif args.step == 'featurize':

		# Read clean dataframe
		features = pd.read_csv(args.input)

		# Create additional features
		features = create_visible_range(features)
		features = create_visible_norm_range(features)
		features = create_log_entropy(features)
		features = create_entropy_x_contrast(features)
		features = create_IR_range(features)
		features = create_IR_norm_range(features)

		# Export updated dataframe
		features.to_csv(args.output)

	elif args.step == 'test':

		# Get list of modeling features based on intersection of 
		# expected and available features
		features_df = pd.read_csv(args.input)
		all_features = config['features']
		available_features = features_df.columns.tolist()
		model_features = list(set(available_features) & set(all_features))

		# Create dataframes/series for features and labels
		X = features_df[model_features]
		y = features_df["class"]

		# Split data into training and testing
		X_train, X_test, y_train, y_test = model_selection.train_test_split(
			X, y, **config['train_test_split'])

		# Fit with training data
		lr = fit_logistic_regression(X_train, y_train, **config['lr_config'])

		# Obtain test metrics
		get_model_metrics(X_test, y_test, lr, args.output)

	elif args.step == 'fit':
		# Get list of modeling features based on intersection of 
		# expected and available features
		features_df = pd.read_csv(args.input)
		all_features = config['features']
		available_features = features_df.columns.tolist()
		model_features = list(set(available_features) & set(all_features))

		# Create dataframes/series for features and labels
		X = features_df[model_features]
		y = features_df["class"]

		# Fit with full data
		lr = fit_logistic_regression(X, y, 
			save_path = args.output, **config['lr_config'])

