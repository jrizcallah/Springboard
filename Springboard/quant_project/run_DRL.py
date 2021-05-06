## This file contains the function that runs all of the other
## functions. It puts the whole thing together.

# libraries
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv
import os

## These ones were made by me
# preprocessor
from preprocessing.preprocessors import *
# configurator
from config.config import *
# model
from model.models import *

def run_model() -> None:
	"""Train the model"""

	# read and preprocess data
	preprocessed_path = "preprocessed_data.csv"
	if os.path.exists(preprocessed_path):
		data = pd.read_csv(preprocessed_path, index_col=0)
	else:
		data = preprocess_data()
		data = add_turbulence(data)
		data.to_csv(preprocessed_path)

	print("Peek at data: /n", data.head())
	print("Size of data: /n", data.size)

	# train through 2015, then validate, then test from 2018 on
	unique_trade_date = data[(data['date'] > "2009-01-01")&(data['date'] <= "2020-12-01")].date.unique()

	# rebalance_window is the number of months to retrain the model
	# validation_window is the number of months to validate the models 
	# and select for trading
	rebalance_window = 63
	validation_window = 63

	## run the ensemble strategy
	run_ensemble_strategy(df = data, 
		unique_trade_date = unique_trade_date,
		rebalance_window = rebalance_window,
		validation_window = validation_window)

if __name__ == "__main__":
	run_model()

