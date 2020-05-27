import argparse
import logging
import yaml
import pandas as pd

from sklearn import model_selection

from src.data_prep import get_raw_data, clean_data, create_features
from src.modeling import fit_logistic_regression, get_model_metrics


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
		data_df = pd.read_csv(args.input)
		create_features(data_df, args.output)

	elif args.step == 'test':
		features_df = pd.read_csv(args.input)
		features = config['features']
		X = features_df[features]
		y = features_df["class"]

		# Split data into training and testing
		X_train, X_test, y_train, y_test = model_selection.train_test_split(
			X, y, **config['train_test_split'])

		# Fit with training data
		lr = fit_logistic_regression(X_train, y_train, **config['lr_config'])

		# Obtain test metrics
		get_model_metrics(X_test, y_test, lr, args.output)

	elif args.step == 'fit':
		features_df = pd.read_csv(args.input)
		features = config['features']
		X = features_df[features]
		y = features_df["class"]

		# Fit with full data
		lr = fit_logistic_regression(X, y, 
			save_path = args.output, **config['lr_config'])

