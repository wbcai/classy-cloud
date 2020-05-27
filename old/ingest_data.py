import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def ingest_data(filename, c1_start_ix, c1_end_ix, c2_start_ix, c2_end_ix):
	
	"""Create dataframe from cloud.data with two cloud classes

	Inputs:
		filename (str): raw cloud data
		*_ix (int): index ranges for each cloud class

	Output:
		data (obj): 'DataFrame' of cloud features and label
	"""


	columns = ['visible_mean', 'visible_max', 'visible_min', 
		   'visible_mean_distribution', 'visible_contrast', 
		   'visible_entropy', 'visible_second_angular_momentum', 
		   'IR_mean', 'IR_max', 'IR_min']

	
	logger.info("Acquiring data from %s", source_url)

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
	if len(first_cloud[0]) == len(columns):
		logger.info("Raw data features all present")
		first_cloud = pd.DataFrame(first_cloud, columns=columns)
		second_cloud = pd.DataFrame(second_cloud, columns=columns)
	else:
		logger.warning("Not all expected features found in raw data")
		first_cloud = pd.DataFrame(first_cloud)
		second_cloud = pd.DataFrame(second_cloud)

	# Append cloud class to dataframes
	first_cloud['class'] = np.zeros(len(first_cloud))
	second_cloud['class'] = np.ones(len(second_cloud))

	# Final DataFrame with both types of clouds
	data = pd.concat([first_cloud, second_cloud])
	
	return data

