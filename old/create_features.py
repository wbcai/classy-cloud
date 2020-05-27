import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_features(features):
	
	"""Create additional features and append to features 'DataFrame' """

	expected_cols = ["visible_max", "visible_min", "visible_mean", "visible_entropy",
				"visible_contrast", "IR_min", "IR_max"]

	input_cols = features.columns.tolist()

	if set(expected_df).issubset(set(input_cols)):
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
	else:
		logger.warning("Not all required columns present; no add'l features generated")
		
	return features