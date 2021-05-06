""" This file defines the environment for validating the DRL models """

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import pickle
matplotlib.use('Agg')

import gym
from gym.utils import seeding
from gym import spaces

## Normalization Factor of 100 Contracts per trade
HMAX_NORMALIZE = 100

# Inital Account Balance or $1 million
INITIAL_ACCOUNT_BALANCE = 1000000

# Number of contracts we can trade
ASSET_DIM = 78

# Transaction fee of 1/1000th of total trade
TRANSACTION_FEE_PERCENT = 0.001

# Reward Scaling Parameter
REWARD_SCALING = 1e-4

class AssetEnvValidation(gym.Env):
	""" Trading Environment for Validation """
	metadata = {'render.modes':['human']}

	def __init__(self, df, day=0, turbulence_threshold=140, iteration=''):

		self.day = day
		self.df = df

		""" Action Space:
				Normalized in [-1, 1]
				Dimension = ASSET_DIM
		"""
		self.action_space = spaces.Box(low=-1, high=1, shape=(ASSET_DIM,))

		""" Observation Space:
				Agent can see:
					Account balance [1]
					Asset Prices [78]
					Number of Each Contract Currently in Portfolio [78]
					MACD [78]
					RSI [78]
					CCI [78]
					ADX [78]
				Total Observation Dim:
					1 + 78 + 78 + 78 + 78 + 78 + 78 = (469,)
		"""
		self.observation_space = spaces.Box(low=0, high=np.inf, shape=(469,))

		# Load data from DataFrame
		self.unique_dates = self.df['date'].unique()
		self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
		self.prices = self.data['close']

		# We don't start at the end
		self.terminal = False

		# Set turbulence threshold (kind of a parameter for risk aversion)
		self.turbulence_threshold = turbulence_threshold

		# Initialize the State
		self.state = [INITIAL_ACCOUNT_BALANCE] + self.data['close_scaled'].values.tolist() + [0]*ASSET_DIM + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

		# Initialze reward
		self.reward = 0

		# Initialize Turbulence
		self.turbulence = 0

		# Initialize transaction costs
		self.cost = 0

		# Initialize count of trades
		self.trades = 0

		# Initialize memory
		self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
		self.rewards_memory = []
		# set seed
		self._seed()

		self.iteration = iteration


	def _sell_asset(self, index, action):
		""" This function performs a SELL action based on action sign"""
		if self.turbulence < self.turbulence_threshold:
			if self.state[index + ASSET_DIM + 1] > 0:
				# update balance
				self.state[0] += self.prices.iloc[index] * min(abs(action), self.state[index + ASSET_DIM + 1]) * (1 - TRANSACTION_FEE_PERCENT)

				# Update holdings
				self.state[index + ASSET_DIM + 1] -= min(abs(action), self.state[index + ASSET_DIM + 1])

				# calculate transaction costs
				self.cost += self.prices.iloc[index] * min(abs(action), self.state[index + ASSET_DIM + 1]) * TRANSACTION_FEE_PERCENT

				# update trade count
				self.trades += 1
			else:
				pass

		else:
			# If turbulence is over the threshold, just close all positions
			if self.state[index + ASSET_DIM + 1] > 0:
				# update balance
				self.state[0] += self.prices.iloc[index] * self.state[index + ASSET_DIM + 1] * (1 - TRANSACTION_FEE_PERCENT)

				# update holdings
				self.state[index + ASSET_DIM + 1] = 0

				# update transaction costs
				self.cost += self.prices.iloc[index] * self.state[index + ASSET_DIM + 1] * TRANSACTION_FEE_PERCENT

				# Update trade count
				self.trades += 1
			else:
				pass



	def _buy_asset(self, index, action):
		"""This function performs a BUY action based on the sign of the action"""
		if self.turbulence < self.turbulence_threshold:
			
			# How many CAN we buy?
			available_amount = self.state[0] // self.prices.iloc[index]

			# Update Balance
			self.state[0] -= self.prices.iloc[index] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

			# update holdings
			self.state[index + ASSET_DIM + 1] += min(available_amount, action)

			# update transaction costs
			self.cost += self.prices.iloc[index] * min(available_amount, action) * TRANSACTION_FEE_PERCENT

			# update trade count
			self.trades += 1
		else:
			pass


	def step(self, actions):
		"""This function steps forward in time to the next state"""
		
		# Are we at the end?
		self.terminal = self.day >= (len(self.df['date'].unique())-1)

		# If this is the end, plot and save performance figures
		if self.terminal:
			print("TERMINAL VALIDATION STATE REACHED")
			plt.plot(self.asset_memory, 'r')
			plt.savefig('results/account_value_validation_{}.png'.format(self.iteration))
			plt.close()

			# save asset memory data
			df_total_value = pd.DataFrame(self.asset_memory)
			df_total_value.to_csv('results/account_value_validation_{}.csv'.format(self.iteration))

			# Get ending balance
			end_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]))

			# get Sharpe ratio
			df_total_value.columns = ['account_value']
			df_total_value['daily_return'] = df_total_value.pct_change(1)
			std_total_value = df_total_value['daily_return'].std()
			if std_total_value > 0:
				sharpe = (4 ** 0.5)*df_total_value['daily_return'].mean() / std_total_value
			else:
				std_total_value = 1
				sharpe = (4 ** 0.5)*df_total_value['daily_return'].mean() / std_total_value

			return self.state, self.reward, self.terminal, {}

		# If we are not at the end, step forward in time
		else:
			actions = actions * HMAX_NORMALIZE

			# If turbulence is too high, just close all positions
			if self.turbulence >= self.turbulence_threshold:
				actions = np.array([-HMAX_NORMALIZE] * ASSET_DIM)

			# get starting account value
			begin_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]))

			# sort actions (index)
			argsort_actions = np.argsort(actions)

			# get index of contracts to sell/buy
			sell_index = argsort_actions[:np.where(actions<0)[0].shape[0]]
			buy_index = argsort_actions[::-1][:np.where(actions>0)[0].shape[0]]

			# take actions
			for index in sell_index:
				self._sell_asset(index, actions[index])

			for index in buy_index:
				self._buy_asset(index, actions[index])

			# Step forward in time
			self.day += 1
			self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
			self.prices = self.data['close']
			self.turbulence = self.data['turbulence'].values[0]

			# get new state
			self.state = [self.state[0]] + self.data['close_scaled'].values.tolist() + list(self.state[(ASSET_DIM + 1): (ASSET_DIM * 2 + 1)]) + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

			# get new total value, used to calculate the reward
			end_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]))	
			
			# add new value to memory
			self.asset_memory.append(end_total_asset)

			# calculate reward
			self.reward = end_total_asset - begin_total_asset

			# add reward to memory
			self.rewards_memory.append(self.reward)

			# Scale the reward
			self.reward = self.reward / begin_total_asset

		return self.state, self.reward, self.terminal, {}


	def reset(self):
		self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
		self.day = 0
		self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
		self.prices = self.data['close']
		self.turbulence = 0
		self.cost = 0
		self.trades = 0
		self.terminal = False
		self.rewards_memory = []

		self.state = [INITIAL_ACCOUNT_BALANCE] + self.data['close_scaled'].values.tolist() + [0]*ASSET_DIM + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

		return self.state

	def render(self, mode='human', close=False):
		return self.state

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

