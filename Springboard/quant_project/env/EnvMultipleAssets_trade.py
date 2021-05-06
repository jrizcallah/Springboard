""" This file defines the Trading Environment for use with OpenAI Gym"""

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import pickle

import gym
from gym.utils import seeding
from gym import spaces

# Normalize trading, 100 shares per trade
HMAX_NORMALIZE = 100

# Initially, we have $1 million
INITIAL_ACCOUNT_BALANCE = 1000000

# We have 75 futures to choose from
ASSET_DIM = 78

# Transaction fees of 1/100th of trade price
TRANSACTION_FEE_PERCENT = 0.001

# Reward scaling
REWARD_SCALING = 1e-4

class AssetEnvTrade(gym.Env):
	"""This environment is for the TRADING portion of the experiment"""
	metadata = {'render.modes':['human']}

	def __init__(self, df, day=0, turbulence_threshold=140, initial=True, previous_state=[], model_name='', iteration=''):

		self.day = day
		self.df = df
		self.initial = initial 
		self.previous_state = previous_state

		"""
		Action space is normalized to be in [-1,1]
			Dimension is ASSET_DIM
		"""
		self.action_space = spaces.Box(low=-1, high=1, shape=(ASSET_DIM,))

		"""
		Observation space
			Agent can see:
				Account balance [1]
				Asset prices [78]
				Number of each contract owned [78]
				MACD [78]
				RSI [78]
				CCI[78]
				ADX [78]

			Observation Dimension:
				1 + 78 + 78 + 78 + 78 + 78 + 78 = 469
		"""
		self.observation_space = spaces.Box(low=0, high=np.inf, shape=(469,))

		# Load data from DataFrame
		self.unique_dates = self.df['date'].unique()
		self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
		self.prices = self.data['close']

		# We do not start at the end
		self.terminal = False

		# Set the turbulence threshold
		self.turbulence_threshold = turbulence_threshold

		# Initialize the state
		self.state = [INITIAL_ACCOUNT_BALANCE] + self.data['close_scaled'].values.tolist() + [0]*ASSET_DIM + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

		# Initialize reward
		self.reward = 0

		# Initialize turbulence
		self.turbulence = 0

		# Initialize trading costs
		self.cost = 0

		# initialize counts of trades
		self.trades = 0

		# initialize memory of assets and rewards
		self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
		self.rewards_memory = []

		# set seed
		self._seed()

		# set model name
		self.model_name = model_name

		self.iteration = iteration


	def _sell_asset(self, index, action):
		"""Performs a SELL action, based on action sign"""
		if self.turbulence < self.turbulence_threshold:
			if self.state[index + ASSET_DIM + 1] > 0:
				# update balance
				self.state[0] += self.prices.iloc[index] * min(abs(action), self.state[index + ASSET_DIM + 1]) * (1 - TRANSACTION_FEE_PERCENT)

				# update holdings
				self.state[index + ASSET_DIM + 1] -= min(abs(action), self.state[index + ASSET_DIM + 1])

				# update transaction costs
				self.cost += self.prices.iloc[index] * min(abs(action), self.state[index + ASSET_DIM + 1]) * TRANSACTION_FEE_PERCENT

				# update trade count
				self.trades += 1
			else:
				pass
		# if turbulence is too high, close all positions
		else:
			if self.state[index + ASSET_DIM + 1] > 0:
				# update balance
				self.state[0] += self.prices.iloc[index] * self.state[index + ASSET_DIM + 1] * (1 - TRANSACTION_FEE_PERCENT)

				# update holdings
				self.state[index + ASSET_DIM + 1] = 0

				# update transaction costs
				self.cost += self.prices.iloc[index] * self.state[index + ASSET_DIM + 1] * TRANSACTION_FEE_PERCENT

				# update trade count
				self.trades += 1
			else:
				pass


	def _buy_asset(self, index, action):
		"""performs a BUY action, based on sign of action"""
		if self.turbulence < self.turbulence_threshold:

			# how many CAN we buy?
			available_amount = self.state[0] // self.prices.iloc[index]

			# update balance
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
		"""This function steps forward in time"""

		# Are we at the end?
		self.terminal = self.day >= (len(self.df['date'].unique())-1)

		# If we are at the end, make and save performance plots and data
		if self.terminal:
			plt.plot(self.asset_memory,'r')
			plt.savefig('results/account_value_trade_{}_{}.png'.format(self.model_name, self.iteration))
			plt.close()

			# get and save final value data
			df_total_value = pd.DataFrame(self.asset_memory)
			df_total_value.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))

			end_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM + 1): (ASSET_DIM * 2 + 1)]))

			print("previous_total_asset:{}".format(self.asset_memory[0]))
			print("end_total_asset:{}".format(end_total_asset))
			print("total_reward:{}".format(self.state[0]+sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)])) - self.asset_memory[0] ))
			print("total_cost:{}".format(self.cost))
			print("total_trades:{}".format(self.trades))

			df_total_value.columns = ['account_value']
			df_total_value['daily_returns'] = df_total_value.pct_change(1)
			sharpe = (4**0.5) * df_total_value['daily_returns'].mean() / df_total_value['daily_returns'].std()

			print("Sharpe Ratio: {}".format(sharpe))


			# get rewards
			df_rewards = pd.DataFrame(self.rewards_memory)
			df_rewards.to_csv('results/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))

			return self.state, self.reward, self.terminal, {}

		# If this is not the end, take actions and get new state
		else:
			actions = actions * HMAX_NORMALIZE

			# if turbulence is too high, sell everything
			if self.turbulence >= self.turbulence_threshold:
				actions = np.array([-HMAX_NORMALIZE] * ASSET_DIM)

			# get starting portfolio value
			begin_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]))

			# sort action index
			argsort_actions = np.argsort(actions)

			# get indices for sell/buy actions
			sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
			buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

			# perform buy and sell actions
			for index in sell_index:
				self._sell_asset(index, actions[index])

			for index in buy_index:
				self._buy_asset(index, actions[index])

			# move forward in time
			self.day += 1
			# get new data
			self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
			self.prices = self.data['close']

			# get new turbulence
			self.turbulence = self.data['turbulence'].values[0]

			# get new state
			self.state = [self.state[0]] + self.data['close_scaled'].values.tolist() + list(self.state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]) + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()

			# get new total portfolio value, to calculate the reward
			end_total_asset = self.state[0] + sum(np.array(self.prices) * np.array(self.state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]))

			# append new value to memory
			self.asset_memory.append(end_total_asset)

			# calculate reward
			self.reward = end_total_asset - begin_total_asset

			# append reward to memory
			self.rewards_memory.append(self.reward)

			# scale the reward
			self.reward = self.reward / begin_total_asset

		return self.state, self.reward, self.terminal, {}


	def reset(self):
		# are we starting over completely?
		if self.initial:
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
		else:
			previous_total_asset = self.previous_state[0] + sum(np.array(self.prices) * np.array(self.previous_state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]))
			self.asset_memory = [previous_total_asset]
			self.day = 0
			self.data = self.df[self.df['date'] == self.unique_dates[self.day]]
			self.prices = self.data['close']
			self.turbulence = 0
			self.cost = 0
			self.trades = 0
			self.terminal = False
			self.rewards_memory = []

			self.state = [self.previous_state[0]] + self.data['close_scaled'].values.tolist() + list(self.previous_state[(ASSET_DIM + 1):(ASSET_DIM * 2 + 1)]) + self.data['macd'].values.tolist() + self.data['rsi'].values.tolist() + self.data['cci'].values.tolist() + self.data['adx'].values.tolist()
		return self.state


	def render(self, mode='human', close=False):
		return self.state

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]