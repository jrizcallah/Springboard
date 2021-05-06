####
Hyperparameters for Ensemble Reinforcement Learning Trading System
####

= = = = Most Recent Version: 5-2-2021
Observation Space: SCALED
	Close Prices: Divided by first close price
	MACD: Not scaled
	RSI: Divided by 100, in [0,1]
	CCI: Standardized to mean=0, s.d.=1
	ADX: Divided by 100, in [0,1]

Action Space: SCALED
	Limited to [-1, 1], then multiplied by 100 to reach number of contracts to buy/sell

	Sell Limit:
		No short selling

	Buy Limit:
		No leverage

Reward Space: SCALED IN TRAINING ONLY
	Training:
		Reward is the percent change in portfolio value from the previous day
	Validation/Testing:
		Reward is difference in portfolio value from the previous day

Rebalance Window: 63
Validation Window: 63

Model Training Hyperparameters:
	A2C:
		Timesteps = 35000
	PPO:
		Timesteps = 100000
	DDPG:
		Timesteps = 10000


Final Value: 1187941.74418836
Annual Return: 1.5%
Total Sharpe: 0.18
Two-Year Sharpe: 1.006
Max Drawdown: -0.25
Pct. Winners: 49.5%
Mean-Win/Mean-Loss Ratio: 1.03:1

 = = = Notes:
 Appears to fit training data, but patterns do not generalize to unseen data. Could try scaling the reward in the validation and testing environments as well? Should also track performance of the three RL algorithms in validation and which one is used for trading, so we can compare performance among them and relationships between validation Sharpe and trading Sharpe.


= = = = Most Profitable Version: 4-19-2021
Observation Space: NOT SCALED
	Close Prices
	MACD
	RSI
	CCI
	ADX

Action Space: SCALED
	Limited to [-1, 1], then multiplied by 100 to reach number of contracts to buy/sell

	Sell Limit:
		No short selling

	Buy Limit:
		No leverage

Reward Space: NOT SCALED
	Training/Validation/Testing:
		Reward is difference in portfolio value from the previous day

Rebalance Window: 63
Validation Window: 63

Model Training Hyperparameters:
	A2C:
		Timesteps = 100000
	PPO:
		Timesteps = 100000
	DDPG:
		Timesteps = 100000

Final Value: 1558610.53005131
Annual Return: 4%
Total Sharpe: 0.41
Two-Year Sharpe: -0.26
Max Drawdown: -0.25
Pct. Winners: 50.2
Mean-Win/Mean-Loss Ratio: 1.05:1

 = = = Notes:
 Solid performance in the in-sample period, but out-of-sample (from 2015) losses are heavy. 