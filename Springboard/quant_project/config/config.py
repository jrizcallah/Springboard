## This file contains the configuration info

import pathlib
import pandas as pd 
import datetime
import os

TRAINING_DATA_FILE = "data/training_data.csv"

todays_date = str(datetime.datetime.now())[:10]
TRAINED_MODEL_DIR = f"trained_models/{todays_date}"
if os.path.isdir(TRAINED_MODEL_DIR):
	os.rmdir(TRAINED_MODEL_DIR)
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA ="data/training_data_turbulence_index.csv"

TESTING_DATA_FILE = "testing_data.csv"