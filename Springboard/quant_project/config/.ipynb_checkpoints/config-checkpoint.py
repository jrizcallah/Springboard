## This file contains the configuration info

import pathlib
import pandas as pd 
import datetime
import os

TRAINING_DATA_FILE = "data/training_data.csv"

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA ="data/training_data_turbulence_index.csv"

TESTING_DATA_FILE = "testing_data.csv"