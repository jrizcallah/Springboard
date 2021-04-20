## This file contains all of the preprocessing functions 
## for the data

import numpy as np 
import pandas as pd 
from stockstats import StockDataFrame as Sdf 
from config import config

import xarray as xr
import qnt.ta as dnta
import qnt.data as qndata


def load_dataset(*, file_name: str) -> pd.DataFrame:
	"""
	load the csv file from path, return as a dataframe
	The data needs to have a single "close" column.
	If there is also an adjusted close column, rename it "close"
	and delete the old one
	"""
	_data = pd.read_csv(file_name, index="date")
	return _data

def load_quantiacs_futures(period: int):
	"""
	load futures data from quantiacs, 
	return dataframe of OHLC data
	"""
	arr = qndata.futures_load_data(tail=period)
	O = arr.sel(field = 'open').to_pandas().reset_index()
	H = arr.sel(field = 'high').to_pandas().reset_index()
	L = arr.sel(field = 'low').to_pandas().reset_index()
	C = arr.sel(field = 'close').to_pandas().reset_index()
	V = arr.sel(field = 'volume').to_pandas().reset_index()

	"""
	There is a known issue in the data, where the F_EB price on
	2020-12-22 is listed as 10.535 instead of 100.535. We will NOT
	fix it here. Please keep the problem in mind when validating
	and testing the trader
	"""

	# melt the dataframe, then change column names to make
	# them easily joinable
	O = pd.melt(O, "time")
	O.columns = ['date', 'asset', 'open']

	H = pd.melt(H, "time")
	H.columns = ["date", "asset", "high"]

	L = pd.melt(L, "time")
	L.columns = ['date', 'asset', 'low']

	C = pd.melt(C, "time")
	C.columns = ["date", "asset", "close"]

	V = pd.melt(V, "time")
	V.columns = ["date", "asset", "close"]

	# join into one big dataframe
	final_df = O.merge(H).merge(L).merge(C).merge(V)
	return final_df

def data_split(df, start, train_end, test_end, test_start=None):
	"""
	Split the data into training and testing periods
	"""
	if test_start == None:
		test_start = train_end
	training_data = df[(df["date"] >= start) & (df["date"] < train_end)]
	testing_data = df[(df['date'] >= test_start) & (df['date'] < test_end)]
	return training_data, testing_data

def add_technical_indicators(df):
	"""
	calculate technical indicators:
		use stockstats package

	Before reaching this function, the DataFrame must have columns like:
		date, asset, open, high, low, close, volume
	"""
	asset_df = Sdf.retype(df.copy())

	unique_assets = asset_df['asset'].unique()

	macd = pd.DataFrame()
	rsi = pd.DataFrame()
	cci = pd.DataFrame()
	adx = pd.DataFrame()

	for i in range(len(unique_assets)):

		# moving average convergence divergence (MACD)
		temp_macd = asset_df[asset_df['asset'] == unique_assets[i]]['macd']
		temp_macd = pd.DataFrame(temp_macd)
		macd = macd.append(temp_macd, ignore_index=True)

		# relative strength index (RSI)
		temp_rsi = asset_df[asset_df['asset'] == unique_assets[i]]['rsi_30']
		temp_rsi = pd.DataFrame(temp_rsi)
		rsi = rsi.append(temp_rsi, ignore_index=True)

		# commodity channel index (CCI)
		temp_cci = asset_df[asset_df['asset'] == unique_assets[i]]['cci_30']
		temp_cci = pd.DataFrame(temp_cci)
		cci = cci.append(temp_cci, ignore_index=True)

		# average directional index (ADX)
		temp_adx = asset_df[asset_df['asset'] == unique_assets[i]]['dx_30']
		temp_adx = pd.DataFrame(temp_adx)
		adx = adx.append(temp_adx, ignore_index=True)

	# attach technical indicators to original DataFrame
	df['macd'] = macd
	df['rsi'] = rsi
	df['cci'] = cci
	df['adx'] = adx

	return df

def preprocess_data(quantiacs=True):
	"""
	This is the data preprocessing pipeline.
	It pieces everything together to get the final data
	"""

	# Do we want quantiacs data or are we using a .csv file?
	if quantiacs = True:
		df = load_quantiacs_futures(period=config.DATA_PERIOD)
	else:
		df = load_dataset(file_name=config.TRAINING_DATA_FILE)

	# Add the technical indicators
	df_final = add_technical_indicators(df)

	# treat the na's
	df_final.fillna(method='bfill', inplace=True)

	return df_final

def add_turbulence(df):
	"""
	Add a turbulence index
	"""
	turbulence_index = calculate_turbulence(df)
	df = df.merge(turbulence_index, on='date')
	df = df.sort_values(["date", "asset"]).reset_index(drop=True)

	return df

def calculate_turbulence(df):
	"""
	This function calculated the turbulence
	"""
	df_price_pivot = df.pivot(index="date", columns="asset", values="close")
	unique_date = df["date"].unique()

	# start after a year
	start = 252
	turbulence_index = [0]*start

	count = 0
	for i in range(start, len(unique_date)):
		current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
		hist_price = df_price_pivot[[n in unique_date[0:i]] for n in df_price_pivot.index]
		cov_temp = hist_price.cov()
		current_temp = (current_price - np.mean(hist_price, axis=0))
		temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
		if temp > 0:
			count += 1
			if count > 2:
				turbulence_temp = temp[0][0]
			else:
				turbulence_temp = 0
		else:
			turbulence_temp = 0
		turbulence_index.append(turbulence_temp)

	turbulence_index = pd.DataFrame({'date':df_price_pivot.index, 
		'turbulence':turbulence_index})

	return turbulence_index



