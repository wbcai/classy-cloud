import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
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

def create_visible_range(features):
	"""Create visible_range feature
		args:
			features (obj): "DataFrame" of features
		returns:
			features (obj): "DataFrame" of features with additional feature
	"""

	# Ensure dependency columns are numeric
	expected_cols = ["visible_max", "visible_min"]
	type_cond = True
	for col in expected_cols:
		if is_numeric_dtype(features[col]) == False:
			type_cond = False
	if type_cond == False:
		logger.warning("visible_range not created; non-numeric dependency column")

	# Ensure all values of visible_max is greater than those of visible_min
	min_max_cond =  sum(features.visible_max < features.visible_min) == 0
	if min_max_cond == False:
		logger.warning("visible_range not created; not all visible_max > visible_min")

	# Create feature if all conditions met
	if type_cond & min_max_cond:
		features['visible_range'] = (features.visible_max - features.visible_min)
		logger.info("visible_range created")

	return features

def create_visible_norm_range(features):
	"""Create visible_norm_range feature
		args:
			features (obj): "DataFrame" of features
		returns:
			features (obj): "DataFrame" of features with additional feature
	"""

	# Ensure dependency columns are numeric
	expected_cols = ["visible_max", "visible_min", "visible_mean"]
	type_cond = True
	for col in expected_cols:
		if is_numeric_dtype(features[col]) == False:
			type_cond = False
	if type_cond == False:
		logger.warning("visible_norm_range not created; non-numeric dependency column")

	# Ensure all values of visible_max is greater than those of visible_min
	min_max_cond = sum(features.visible_max < features.visible_min) == 0
	if min_max_cond == False:
		logger.warning("Feature visible_norm_range not created; not all visible_max > visible_min")

	# Ensure mean values are not 0
	mean_cond = sum(features.visible_mean == 0) == 0
	if mean_cond == False:
		logger.warning("Feature visible_norm_range not created; visible_mean cannot have 0 values")

	# Create feature if all conditions met
	if type_cond & min_max_cond & mean_cond:
		features['visible_norm_range'] = (
			features.visible_max - features.visible_min).divide(features.visible_mean)
		logger.info("visible_norm_range created")

	return features

def create_log_entropy(features):
	"""Create log_entropy feature
		args:
			features (obj): "DataFrame" of features
		returns:
			features (obj): "DataFrame" of features with additional feature
	"""

	# Ensure dependency column is numeric
	type_cond = is_numeric_dtype(features['visible_entropy'])
	if type_cond == False:
		logger.warning("log_entropy not created; non-numeric visible_entropy column")

	# Ensure dependency column is not less than or equal to 0
	zero_cond = len(features.loc[features.visible_entropy <= 0]) == 0
	if zero_cond == False:
		logger.warning("log_entropy not created; value(s) <= 0 found in visible_entropy")

	# Create feature if all conditions met
	if type_cond & zero_cond:
		features['log_entropy'] = features.visible_entropy.apply(np.log)
		logger.info("log_entropy created")

	return features

def create_entropy_x_contrast(features):
	"""Create entropy_x_contrast feature
		args:
			features (obj): "DataFrame" of features
		returns:
			features (obj): "DataFrame" of features with additional feature
	"""

	# Ensure all dependency columns are numeric
	expected_cols = ["visible_contrast", "visible_entropy"]
	type_cond = True
	for col in expected_cols:
		if is_numeric_dtype(features[col]) == False:
			type_cond = False
	if type_cond == False:
		logger.warning("entropy_x_contrast not created; non-numeric dependency column")

	# Create feature if condition met
	if type_cond:
		features['entropy_x_contrast'] = features.visible_contrast.multiply(
			features.visible_entropy)

	return features

def create_IR_range(features):
	"""Create IR_range feature
		args:
			features (obj): "DataFrame" of features
		returns:
			features (obj): "DataFrame" of features with additional feature
	"""

	# Ensure dependency columns are numeric
	expected_cols = ["IR_max", "IR_min"]
	type_cond = True
	for col in expected_cols:
		if is_numeric_dtype(features[col]) == False:
			type_cond = False
	if type_cond == False:
		logger.warning("IR_range not created; non-numeric dependency column")

	# Ensure all values of IR_max is greater than those of IR_min
	min_max_cond =  sum(features.IR_max < features.IR_min) == 0
	if min_max_cond == False:
		logger.warning("IR_range not created; not all IR_max > IR_min")

	# Create feature if all conditions met
	if type_cond & min_max_cond:
		features['IR_range']  = features.IR_max - features.IR_min
		logger.info("IR_range created")

	return features

def create_IR_norm_range(features):
	"""Create IR_norm_range feature
		args:
			features (obj): "DataFrame" of features
		returns:
			features (obj): "DataFrame" of features with additional feature
	"""

	# Ensure dependency columns are numeric
	expected_cols = ["IR_max", "IR_min", "IR_mean"]
	type_cond = True
	for col in expected_cols:
		if is_numeric_dtype(features[col]) == False:
			type_cond = False
	if type_cond == False:
		logger.warning("IR_norm_range not created; non-numeric dependency column")

	# Ensure all values of visible_max is greater than those of visible_min
	min_max_cond = sum(features.IR_max < features.IR_min) == 0
	if min_max_cond == False:
		logger.warning("Feature IR_norm_range not created; not all IR_max > IR_min")

	# Ensure mean values are not 0
	mean_cond = sum(features.IR_mean == 0) == 0
	if mean_cond == False:
		logger.warning("Feature IR_norm_range not created; IR_mean cannot have 0 values")

	# Create feature if all conditions met
	if type_cond & min_max_cond & mean_cond:
		features['visible_norm_range'] = (
			features.IR_max - features.IR_min).divide(features.IR_mean)
		logger.info("visible_normal_range created")

	return features

