import numpy as np
import pandas as pd
import logging
import requests

logger = logging.getLogger(__name__)

def get_raw_data(url, save_path):

	"""Get raw data via URL"""

	r = requests.get(url, allow_redirects=True)
	open(save_path, 'wb').write(r.content)

	return None

def clean_data(filename, save_path, c1_start_ix, c1_end_ix, 
	c2_start_ix, c2_end_ix, column_names):
	
	"""Create dataframe from cloud.data with two cloud classes

	Args:
		filename (str): raw cloud data
		save_path (str): path to save DataFrame
		**kwargs: index ranges for each cloud class and column names
	Returns:
		data (obj): 'DataFrame' of cloud features and label
	"""
	
	logger.info("Importing data from %s", filename)

	# Parse data file
	with open(filename,'r') as f:
		data = [[s for s in line.split(' ') if s!=''] for line in f.readlines()]
	
	# Split raw data into each cloud type
	first_cloud = data[c1_start_ix:c1_end_ix]
	first_cloud = [[float(s.replace('/n', '')) for s in cloud]
					for cloud in first_cloud]

	second_cloud = data[c2_start_ix:c2_end_ix]
	second_cloud = [[float(s.replace('/n', '')) for s in cloud]
					for cloud in second_cloud]

	# Check that raw data still has same number of features and create dataframes
	if len(first_cloud[0]) == len(column_names):
		logger.info("Raw data features all present")
		first_cloud = pd.DataFrame(first_cloud, columns=column_names)
		second_cloud = pd.DataFrame(second_cloud, columns=column_names)
	else:
		logger.warning("Not all expected features found in raw data")
		first_cloud = pd.DataFrame(first_cloud)
		second_cloud = pd.DataFrame(second_cloud)

	# Append cloud class to dataframes
	first_cloud['class'] = np.zeros(len(first_cloud))
	second_cloud['class'] = np.ones(len(second_cloud))

	# Final DataFrame with both types of clouds
	data = pd.concat([first_cloud, second_cloud])

	data.to_csv(save_path)

	logger.info("Dataframe created and saved to %s", save_path)
	
	return None

def create_features(features, save_path):
	
	"""Create additional features and append to features 'DataFrame' """

	expected_cols = ["visible_max", "visible_min", "visible_mean", "visible_entropy",
				"visible_contrast", "IR_min", "IR_max"]

	input_cols = features.columns.tolist()

	if set(expected_cols).issubset(set(input_cols)):
		logger.info("Creating additional features")
	
		features['visible_range'] = (features.visible_max - features.visible_min)
		
		features['visible_norm_range'] = (
			features.visible_max - features.visible_min).divide(features.visible_mean)

		features['log_entropy'] = features.visible_entropy.apply(np.log)

		features['entropy_x_contrast'] = features.visible_contrast.multiply(
			features.visible_entropy)

		features['IR_range']  = features.IR_max - features.IR_min

		features['IR_norm_range'] = (features.IR_max - features.IR_min).divide(
			features.IR_mean)

		features.to_csv(save_path)

		logger.info("Dataframe with additional features saved to %s", save_path)

	else:
		logger.warning("Not all required columns present; no add'l features generated")
		
	return None
