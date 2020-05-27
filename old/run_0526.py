import argparse
import logging
import yaml

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

	with open(args.config, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	if args.input is not None:
		input = pd.read_csv(args.input)
		logger.info('Input data loaded from %s', args.input)

	# Paths based on step arguement	
	if args.step == 'acquire':
		if args.input is None:
			url = config['acquire_data']['url']
		else:
			url = args.input
		if args.output is None:
			path = config['acquire_data']['save_path']
		else:
			path = args.output
		get_raw_data(url, path)

	elif args.step == 'clean':

		clean_data(config['acquire_data']['save_path'], 
			config['clean_data']['class1_start_ix'], config['clean_data']['class1_end_ix'],
			config['clean_data']['class2_start_ix'], config['clean_data']['class2_end_ix'],
			args.output)
	elif args.step == 'featurize':
		create_features(input, args.output)
	elif args.step == 'test':
		# List of feature names
		features = config['features']
		X = input[features]
		y = input["class"]

		# Split data into training and testing
		X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, **config['train_test_split'])

		# Fit with training data
		lr = fit_logistic_regression(X_train, y, **config['lr_config'])

		# Obtain test metrics
		get_model_metrics(X_test, y_test, lr, config['metrics_dir'])

	elif arg.step == 'fit':
		# List of feature names
		features = config['features']
		X = input[features]
		y = input["class"]

		# Fit with full data
		lr = fit_logistic_regression(X_train, y, 
			save_path = config['final_model_path'], **config['lr_config'])


