""" This file defines the training environment with multiple assets """

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use('Agg')

import gym
from gym.utils import seeding
from gym import spaces


# Shares normalization factor: 100 Shares per Trade
HMAX_NORMALIZE = 100

# Inital Balance of $1 million
INITIAL_ACCOUNT_BALANCE=1000000

# Total Number of Assets that can be bought
ASSET_DIM = 78

# Transaction fees of 1/1000th of trade size
TRANSACTION_FEE_PERCENT = 0.001

# Reward Scaling
REWARD_SCALING = 1e-4

class AssetEnvTrain(gym.Env):
	"""A Trading Environment for OpenAI gym"""
	metadata = {'render.modes':['human']}

	def __init__(self, df, day=0):
		self.day = day
		self.df = df

		"""
		Create the action space.
		Make it normalized between -1 and 1, with STOCK_DIM
		"""
		self.action_space = spaces.Box(low=-1, high=1, shape=(ASSET_DIM,))

		"""
		Create observation space.
		Agent can see: 
			Money in account [1], prices for contracts [78],
			owned shares of each contract [78], macd for assets [78],
			rsi for assets [78], cci for assets [78], adx for assets [78]

		Observation space dimension = 1 + 78 + 78 + 78 + 78 + 78 + 78
		Observation space dimension = (469,)
		"""
		self.observation_space = spaces.Box(low=0, high=np.inf, shape=(469,))

		# Load data from pandas dataframe
		self.unique_dates = self.df['date'].unique()
		self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
		self.prices = self.data['close']

		# Terminal is FALSE until the end
		self.terminal = False

		# Initialize the state
		self.state = [INITIAL_ACCOUNT_BALANCE] + self.data['close_scaled'].values.tolist() + [0]*ASSET_DIM + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

		# Initialize the reward
		self.reward = 0
		self.cost = 0

		# Initialize memory
		self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
		self.rewards_memory = []
		self.trades = 0

		# Initialize seed
		self._seed()


	def _sell_asset(self, index, action):
		""" This function performs a sell action based on sign of the action """
		if self.state[index+ASSET_DIM+1] > 0:
			# Update balance
			self.state[0] += self.prices.iloc[index]*min(abs(action), self.state[index + ASSET_DIM + 1]) * (1 - TRANSACTION_FEE_PERCENT)

			self.state[index + ASSET_DIM + 1] -= min(abs(action), self.state[index + ASSET_DIM + 1])
			self.cost += self.prices.iloc[index] * min(abs(action), self.state[index + ASSET_DIM + 1]) * TRANSACTION_FEE_PERCENT

			self.trades += 1
		else:
			pass

	def _buy_asset(self, index, action):
		""" This function performs a buy action based on the sign of the action """
		
		# How many CAN we buy?
		available_amount = self.state[0] // self.prices.iloc[index]

		# Update balance
		self.state[0] -= self.prices.iloc[index] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

		self.state[index + ASSET_DIM + 1] += min(available_amount, action)

		self.cost += self.prices.iloc[index] * min(available_amount, action) * TRANSACTION_FEE_PERCENT

		self.trades += 1

	def step(self, actions):
		""" This function moves the agent forward in time """
		
		# Is this the end?
		self.terminal = self.day >= (len(self.df['date'].unique())-1)

		# If it is the end, then plot and save figures
		if self.terminal:
			plt.plot(self.asset_memory, 'r')
			plt.savefig('results/account_value_train.png')
			plt.close()

			end_total_asset = self.state[0] + sum(np.array(self.prices)*np.array(self.state[(ASSET_DIM+1):(ASSET_DIM*2+1)]))

			df_total_value = pd.DataFrame(self.asset_memory)
			df_total_value.to_csv('results/account_value_train.csv')
			df_total_value.columns = ['account_value']
			df_total_value['daily_return'] = df_total_value.pct_change(1)
			sharpe = (252**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()

			df_rewards = pd.DataFrame(self.rewards_memory)
			df_rewards.to_csv('results/rewards_value_train.csv')

			return self.state, self.reward, self.terminal, {}

		# If it isn't the end, then take actions and move forward in time
		else:

			# Get actions
			actions = actions * HMAX_NORMALIZE

			# Find starting account value
			begin_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM+1):(ASSET_DIM*2+1)]))


			# Take actions
			argsort_actions = np.argsort(actions)

			sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
			buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

			for index in sell_index:
				self._sell_asset(index, actions[index])

			for index in buy_index:
				self._buy_asset(index, actions[index])


			# Move forward in time
			self.day += 1
			self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
			self.prices = self.data['close']

			# Get next state
			self.state = [self.state[0]] + self.data['close_scaled'].values.tolist() + list(self.state[(ASSET_DIM+1):(ASSET_DIM*2+1)]) + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

			# Get new account value
			end_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM+1):(ASSET_DIM*2+1)]))

			# Remember new portfolio value
			self.asset_memory.append(end_total_asset)

			# Calculate Reward
			self.reward = ((end_total_asset - begin_total_asset) / begin_total_asset)

			# Remember reward
			self.rewards_memory.append(self.reward)


			## Standardize Reward
			#mean_reward = np.mean(self.rewards_memory)
			#if len(self.rewards_memory) > 1:
			#	std_reward = np.std(self.rewards_memory)
			#else:
			#	std_reward = 1

			self.reward = (self.reward - 1) * 100


		return self.state, self.reward, self.terminal, {}


	def reset(self):
		self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
		self.day = 0
		self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
		self.prices = self.data['close']
		self.cost = 0
		self.trades = 0
		self.terminal = False
		self.rewards_memory = []

		self.state = [INITIAL_ACCOUNT_BALANCE] + self.data['close_scaled'].values.tolist() + [0]*ASSET_DIM + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

		return self.state


	def render(self, mode='human'):
		return self.state

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
