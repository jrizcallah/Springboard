## This file contains functions to build the actual models

# common libraries
import pandas as pd 
import numpy as np
import time
import gym

# stable_baselines RL stuff
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan


# Custom Functions
from preprocessing.preprocessors import *
from config import config

# Custom Environments
from env.EnvMultipleAssets_train import AssetEnvTrain
from env.EnvMultipleAssets_validation import AssetEnvValidation
from env.EnvMultipleAssets_trade import AssetEnvTrade

"""
Now the model-training functions!
"""

def train_A2C(env_train, model_name, timesteps=25000):
	""" Train the Advantage Actor Critic (A2C) model """

	start = time.time()
	model = A2C('MlpPolicy', env_train, verbose=0)
	model.learn(total_timesteps = timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training Time (A2C): ', (end-start)/60, ' Minutes')

	return model

def train_ACER(env_train, model_name, timesteps=25000):
	""" Train the Actor-Critic with Experience Replay (ACER) model """

	start = time.time()
	model = ACER('MlpPolicy', env_train, verbose=0)
	model.learn(total_timesteps = timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training Time (ACER): ', (end-start)/60, ' Minutes')

	return model

def train_DDPG(env_train, model_name, timesteps = 10000):
	"""Train the Deep Deterministic Policy Gradient model"""

	# Add noise for the DDPG model
	n_actions = env_train.action_space.shape[-1]
	param_noise = None 
	action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))

	# Train the model
	start = time.time()
	model = DDPG('MlpPolicy', env_train, param_noise = param_noise, action_noise = action_noise)
	model.learn(total_timesteps = timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training Time (DDPG): ', (end-start)/60, ' Minutes')

	return model

def train_PPO(env_train, model_name, timesteps=500000):
	"""Train the Proximal Policy Optimization model"""

	start = time.time()
	model = PPO2('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8)
	model.learn(total_timesteps = timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training Time (PPO): ', (end-start)/60, ' Minutes')
	return model

def train_GAIL(env_train, model_name, timesteps=1000):
	"""Train the Generative Adversarial Imitation Learning model"""


	start = time.time()

	## Generate Expert Trajectories
	model = SAC('MlpPolicy', env_train, verbose=1)
	generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

	## Load Expert Dataset
	dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
	
	model = GAIL('MlpPolicy', env_train, dataset, verbose=1)
	model.learn(total_timesteps = timesteps)
	end = time.time()

	model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
	print('Training Time (GAIL): ', (end-start)/60, ' Minutes')

	return model

def get_split_data(df, i, rebalance_window, validation_window, unique_trade_date):
    trade_data = data_split_single(df, start=unique_trade_date[i - rebalance_window], end = unique_trade_date[i])
    train_data, validation_data = data_split(df, start="2009-01-01", train_end = unique_trade_date[i - rebalance_window - validation_window+1], test_end = unique_trade_date[i - rebalance_window])

    train_data['close_scaled'] = train_data['close'].copy()
    validation_data['close_scaled'] = validation_data['close'].copy()
    trade_data['close_scaled'] = trade_data['close'].copy()
    
    ## SCALE THE DATA!
    unique_assets = train_data['asset'].unique()
    for asset in unique_assets:
        #training    
        temp_train = train_data.loc[train_data['asset'] == asset,:].copy()
        train_start = temp_train['close'].iloc[0]
        temp_train.loc[:,'close_scaled'] = temp_train['close_scaled'] / train_start
        temp_train[['rsi', 'adx']] = temp_train[['rsi', 'adx']]/100
        
        train_cci_mean = temp_train['cci'].mean()
        train_cci_std = temp_train['cci'].std()
        temp_train['cci'] = (temp_train['cci'] - train_cci_mean) / train_cci_std
        train_data.loc[train_data['asset'] == asset, :] = temp_train.copy()
        train_data = train_data.fillna(method='ffill').fillna(0)
        
        # validation
        temp_validation = validation_data.loc[validation_data['asset'] == asset,:].copy()
        temp_validation.loc[:,'close_scaled'] = temp_validation['close_scaled'] / train_start
        temp_validation[['rsi', 'adx']] = temp_validation[['rsi', 'adx']]/100

        temp_validation['cci'] = (temp_validation['cci'] - train_cci_mean) / train_cci_std
        validation_data.loc[validation_data['asset'] == asset, :] = temp_validation.copy()
        validation_data = validation_data.fillna(method='ffill').fillna(0)

        # trading
        temp_trade = trade_data.loc[trade_data['asset'] == asset,:].copy()
        temp_trade.loc[:,'close_scaled'] = temp_trade['close_scaled'] / train_start
        temp_trade[['rsi', 'adx']] = temp_trade[['rsi', 'adx']]/100

        temp_trade['cci'] = (temp_trade['cci'] - train_cci_mean) / train_cci_std
        trade_data.loc[trade_data['asset'] == asset, :] = temp_trade.copy()
        trade_data = trade_data.fillna(method='ffill').fillna(0)

    return train_data, validation_data, trade_data


def DRL_prediction(trade_data, model, name, last_state, iter_num, unique_trade_date, rebalance_window, turbulence_threshold, initial):
	"""This function makes predictions using the trained models"""

	## Prepare Trading Environment

	env_trade = DummyVecEnv([lambda: AssetEnvTrade(trade_data, turbulence_threshold=turbulence_threshold, initial=initial, previous_state=last_state, model_name=name, iteration=iter_num)])
	env_trade = VecCheckNan(env_trade, raise_exception=True)
	obs_trade = env_trade.reset()

	trade_length = len(trade_data.index.unique())
	for i in range(trade_length):
		action, _states = model.predict(obs_trade)
		obs_trade, reward, dones, info = env_trade.step(action)

		# If this is the last state...
		if i == (trade_length - 2):
			last_state = env_trade.render()

	# Save the last state, since it includes all relevant performance information
	df_last_state = pd.DataFrame({'last_state':last_state})
	df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
	
	return last_state

def DRL_validation(model, test_data, test_env, test_obs) -> None:
	"""This function runs the validation part of model selection"""
	validation_length = len(test_data['date'].unique())
	for i in range(validation_length):
		action, _states = model.predict(test_obs)
		test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
	"""Calculate Sharpe Ratio from Validation Results"""
	df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
	df_total_value.columns = ['account_value_train']
	df_total_value['daily_return'] = df_total_value.pct_change(1)
	
	sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()

	return sharpe

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
	"""This Ensemble Strategy combines PPO, A2C, and DDPG"""
	print(" = = = = = = START ENSEMBLE STRATEGY = = = = = = ")

	""" 
	The last state of the previous model must be fed to the
	new model as the initial state
	"""
	last_state_ensemble = []

	ppo_sharpe_list = []
	ddpg_sharpe_list = []
	a2c_sharpe_list = []

	model_use = []

	insample_turbulence = df[(df['date'] >= '2009-01-01') & (df['date'] < '2015-01-01')]
	insample_turbulence = insample_turbulence.drop_duplicates(subset=['date'])
	insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.9)

	start = time.time()

	for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
		print(" = = = = = = = = = = = = = ")

		# Start with an empty state
		if i - rebalance_window - validation_window == 0:
			initial = True 
		else:
			initial = False 

		# Tune turbulence based on historical data
		# Use a 1Q turbulence lookback window
		end_date_index = df.index[df['date'] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
		start_date_index = end_date_index - validation_window*30 + 1

		historical_turbulence = df.loc[start_date_index:(end_date_index +1), :]
		historical_turbulence = historical_turbulence.drop_duplicates(subset=['date'])
		historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

		if historical_turbulence_mean > insample_turbulence_threshold:
			"""
			If the historical turbulence mean is greater than the
			in-sample turbulence threshold (the 90% quantile of in-sample turbulence)
			then assume that the market is volatile and  set the
			turbulence threshold to the insample_turbulence_threshold
			"""
			turbulence_threshold = insample_turbulence_threshold
		else:
			"""
			If the historical turbulence mean is NOT greater than the
			in-sample turbulence threshold, then we assume the market
			is less volatile (less risky), and so we turn up the 
			turbulence threshold
			"""
			turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.95)
		print("turbulence_threshold: ", turbulence_threshold)


		## NOW WE CAN SET UP THE ENVIRONMENTS
		train_data, validation_data, trade_data = get_split_data(df=df, i=i, rebalance_window=rebalance_window, validation_window=validation_window, unique_trade_date=unique_trade_date)


		env_train = DummyVecEnv([lambda: AssetEnvTrain(train_data)])
		env_train = VecCheckNan(env_train, raise_exception=True)

		validation_data = validation_data.reset_index(drop=True)

		env_val = DummyVecEnv([lambda: AssetEnvValidation(validation_data, turbulence_threshold=turbulence_threshold, iteration=i)])
		env_val = VecCheckNan(env_val, raise_exception=True)

		obs_val = env_val.reset()
		## END OF ENVIRONMENT SETUP

		## START TRAINING AND VALIDATION

		print(' = = = = = = Model Training From: 2009-01-01 to ', unique_trade_date[i - rebalance_window - validation_window])

		# A2C first
		print(' = = = A2C Training = = = ')
		model_a2c = train_A2C(env_train, model_name="A2C_{}".format(i), timesteps=30000)
		print(' = = = A2C Validation from ', unique_trade_date[i - rebalance_window - validation_window + 1], " to ", unique_trade_date[i - rebalance_window])
		DRL_validation(model=model_a2c, test_data=validation_data, test_env=env_val, test_obs=obs_val)
		sharpe_a2c = get_validation_sharpe(i)
		print("A2C Sharpe Ratio: ", sharpe_a2c)

		# Then ACER
		#print(' = = = ACER Training = = = ')
		#model_acer = train_ACER(env_train, model_name="ACER_{}".format(i), timesteps=100000)
		#print(' = = = ACER Validation from ', unique_trade_date[i - rebalance_window - validation_window + 1], " to ", unique_trade_date[i - rebalance_window])
		#DRL_validation(model=model_acer, test_data=validation_data, test_env=env_val, test_obs=obs_val)
		#sharpe_acer = get_validation_sharpe(i)
		#print("ACER Sharpe Ratio: ", sharpe_acer)

		# Then PPO
		print(' = = = PPO Training = = = ')
		model_ppo = train_PPO(env_train, model_name="PPO_{}".format(i), timesteps=100000)
		print(' = = = PPO Validation from ', unique_trade_date[i - rebalance_window - validation_window + 1], " to ", unique_trade_date[i - rebalance_window])
		DRL_validation(model=model_ppo, test_data=validation_data, test_env=env_val, test_obs=obs_val)
		sharpe_ppo = get_validation_sharpe(i)
		print("PPO Sharpe Ratio: ", sharpe_ppo)


		# Now DDPG
		print(' = = = DDPG Training = = = ')
		model_ddpg = train_DDPG(env_train, model_name="DDPG_{}".format(i), timesteps=10000)
		print(' = = = DDPG Validation from ', unique_trade_date[i - rebalance_window - validation_window + 1], " to ", unique_trade_date[i - rebalance_window])
		DRL_validation(model=model_ddpg, test_data=validation_data, test_env=env_val, test_obs=obs_val)
		sharpe_ddpg = get_validation_sharpe(i)
		print("DDPG Sharpe Ratio: ", sharpe_ddpg)

		# Fanilly, GAIL
		#print(' = = = GAIL Training = = = ')
		#model_gail = train_GAIL(env_train, model_name="GAIL_{}".format(i), timesteps=100000)
		#print(' = = = GAIL Validation from ', unique_trade_date[i - rebalance_window - validation_window + 1], " to ", unique_trade_date[i - rebalance_window])
		#DRL_validation(model=model_gail, test_data=validation_data, test_env=env_val, test_obs=obs_val)
		#sharpe_gail = get_validation_sharpe(i)
		#print("GAIL Sharpe Ratio: ", sharpe_a2c)


		a2c_sharpe_list.append(sharpe_a2c)
		ppo_sharpe_list.append(sharpe_ppo)
		ddpg_sharpe_list.append(sharpe_ddpg)


		# Select model based on Sharpe ratio
		if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
			model_type = 'PPO'
			model_use.append('PPO')
			model_ensemble = model_ppo
		elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
			model_type = 'A2C'
			model_use.append('A2C')
			model_ensemble = model_a2c
		else:
			model_type = 'DDPG'
			model_use.append('DDPG')
			model_ensemble = model_ddpg

		## END OF  TRAINING AND VALIDATION

		## TRAIN BEST MODEL ON WHOLE DATASET BEFORE TRADING
		#whole_training_data = train_data.append(validation_data)

    	## TRAIN WHOLE MODEL	
		#whole_training_env = DummyVecEnv([lambda: AssetEnvTrain(whole_training_data)])
		#whole_training_env = VecCheckNan(whole_training_env, raise_exception=True)
		#print("TRAINING {} MODEL FROM ".format(model_type), whole_training_data['date'].unique().min(), " TO ", whole_training_data['date'].unique().max())
		#if model_type == 'PPO':
		#	model_ensemble = train_PPO(whole_training_env, model_name="whole_PPO_{}".format(i), timesteps=100000)
		#if model_type == 'DDPG':
		#	model_ensemble =  train_DDPG(whole_training_env, model_name="whole_DDPG_{}".format(i), timesteps=100000)
		#if model_type == 'A2C':
		#	model_ensemble = train_A2C(whole_training_env, model_name="A2C_{}".format(i), timesteps=100000)
		#print(" = = = = = WHOLE MODEL TRAINED = = = = =")

		## NOW THE TRADING STARTS
		print(' = = = = = = Trading From: ', unique_trade_date[i - rebalance_window], " to ", unique_trade_date[i], ' = = = = = = ')

		last_state_ensemble = DRL_prediction(trade_data=trade_data, model=model_ensemble, name='ensemble', last_state=last_state_ensemble, iter_num=i, unique_trade_date=unique_trade_date, rebalance_window=rebalance_window, turbulence_threshold=turbulence_threshold, initial=initial)
		## End of Trading

	end = time.time()
	print("Ensemble Strategy Took: ", (end-start)/60, " Minutes")

