import sklearn
from sklearn import linear_model
from sklearn import metrics

import logging
import pickle
from pathlib import Path
import pandas as pd 

logger = logging.getLogger(__name__)

def fit_logistic_regression(X, y, save_path = None, **kwargs):

	""" Fit logistic regression model
	Args:
		X (Obj): 'DataFrame' of features
		y (array): labels
		save_path (str): path to pickle and save model
		**kwargs: Keyword arguments for sklearn.linear_model.LogisticRegression. Please see 
		sklearn documentation for all possible options:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
	 
	Returns:
		model (`sklearn.linear_model.LogisticRegression`): trained model object
	"""

	lr = linear_model.LogisticRegression(**kwargs)

	lr.fit(X, y)

	logger.info("Logistic Regression fitted")

	if save_path is not None:
		#root = Path(".")
		with open(save_path, "wb") as f:
			pickle.dump(lr, f)
		logger.info("Trained model object saved to %s", save_path)
		return None
	else:
		return lr

def get_model_metrics(X_test, y_test, model, save_dir):

	""" Obtain model metrics

	Args:
		X_test (obj): 'DataFrame' of test features
		y_test (array): test labels
		model (`sklearn model`): trained model object
		save_dir (str): directory to save model metrics

	Returns: None

	"""

	# Make predictions with test data
	ypred_proba_test = model.predict_proba(X_test)[:,1]
	ypred_bin_test = model.predict(X_test)

	# Calculate metrics
	auc = sklearn.metrics.roc_auc_score(y_test, ypred_proba_test)
	confusion = sklearn.metrics.confusion_matrix(y_test, ypred_bin_test)
	accuracy = sklearn.metrics.accuracy_score(y_test, ypred_bin_test)
	classification_report = sklearn.metrics.classification_report(y_test, ypred_bin_test)

	confusion_df = pd.DataFrame(confusion,
		index=['Actual negative','Actual positive'],
		columns=['Predicted negative', 'Predicted positive'])

	# Save AUC and accuracy in text file
	text_file = open(save_dir + "auc_accuracy.txt", "w")
	text_file.write("AUC: {} \n".format(round(auc,3)))
	text_file.write("Accuracy: {} \n\n".format(round(accuracy,3)))
	text_file.close()

	# Save classification report in text file
	text_file = open(save_dir + "class_report.txt", "w")
	text_file.write(classification_report)
	text_file.close()

	# Save confusion matrix as csv
	confusion_df.to_csv(save_dir + "confusion.csv")

	logger.info("Model metrics saved in %s", save_dir)

	return None



