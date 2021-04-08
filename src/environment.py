import numpy as np
import pandas as pd
import gym
import copy
import matplotlib.pyplot as plt
import datetime as dt

START_DATE = pd.to_datetime('2012-01-01')
EPS = 1e-8

class DataGenerator:
    '''Provides data for each new episode'''

    def __init__(self, prices, tickers, num_steps, window_size, start_idx, start_date = None):
        '''
        args:
            prices: [n_, t_, 1] vector of prices
            tickers: list of ticker strings
            steps: total number of steps to simulate
            window_size = observation window < 50
            start_date = starting date for generating data
        '''
        self.num_steps = num_steps + 1
        self.window_size = window_size
        self.start_idx = start_idx
        self.start_date = start_date
        #deep copy
        self.data = prices.copy()
        self.tickers = tickers
        self.reset()
    def _step(self):
        #obtain observations for prices
        self.step += 1
        obs = self.data.iloc[self.step: self.step + self.window_size,:].copy()
        ground_truth = self.data.iloc[self.step + self.window_size: self.step + self.window_size + 1,:].copy()
        done = self.step >= self.num_steps
        return obs, done, ground_truth

    def reset(self):
        self.step = 0

        #sample the date
        if self.start_date is None:
            self.idx = np.random.randint(self.window_size, high = self.data.shape[0] - self.num_steps)
        else:
            #get the index corresponding to start_date
            self.idx = date_to_index(self.start_date, START_DATE) - self.start_idx
            data = self.data.iloc[:, self.idx - self.window_size:self.idx + self.num_steps + 1]
            self.data = data
            s1 = self.data.iloc[:, self.step:self.step + self.window_size].copy()
            s2 = self.data.iloc[:, self.step + self.window_size:self.step + self.window_size + 1, :].copy()

            return s1, s2

class PortfolioSim:
    def __init__(self, tickers, num_steps, transaction_cost = 0.001):
        '''
        Portfolio allocation simulation
        args:
            transaction_cost = 0.001 arbitrary
        Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
        '''
        self.tickers = tickers
        self.transaction_cost = transaction_cost
        self.num_steps = num_steps
        self.reset()
    def _step(self, w_1, y_1):
        '''
        Step function
        w_1: new action as a list of portfolio weights
        y_1: price relative vector in returns
        
        Numbered equations from [Jiang 2017]
        '''

        assert w_1.shape == y_1.shape, 'w and r need to have the same shape'


        p_0 = self.p_0
        dw_1 = (y_1 * w_1) / (np.dot(y_1, w_1) + EPS) #eq 7
        mu_1 = self.transaction_cost * (np.abs(dw_1 - w_1)).sum() #eq 16
        assert mu_1 < 1

        p_1 = p_0 * (1 - mu_1) * np.dot(y_1, w_1) #eq 11

        rho_1 = p_1 / p_0 - 1 #returns

        r_1 = np.log((p_1 + EPS) / (p_0 + EPS)) #log returns
        reward = r_1 / self.num_steps * 1000. #eq 22
        self.p_0 = p_1

        #if you run out you're done
        done = p_1 == 0

        info = {
            "reward": reward,
            "log_return": r_1,
            "portfolio_value": p_1,
            "return": y_1.mean(),
            "rate_of_return": rho_1,
            "weights_mean": w_1.mean(),
            "weights_std": w_1.std(),
            "cost": mu_1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p_0 = 1.

class RLEnvironment(gym.Env):
    '''
    Environment for the openAI gym package that serves as a base environment for training different RL frameworks
    for the portfolio allocation problem.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    '''

    def __init__(self, price_history, tickers, start_idx = 0, sample_start_date = None,
                 num_steps = 730, tcost = 0.001,
                 leverage = 1, allow_short = False, window_size = 22):
        self.window_size = window_size
        self.n_ = price_history.shape[0]
        self.start_idx = start_idx
        self.allow_short = allow_short

        self.src_data = DataGenerator(price_history, tickers, num_steps, window_size, start_idx,sample_start_date)

        self.sim_data = PortfolioSim(tickers,num_steps,tcost)
        self.infos=[]

        #openai gym attributes
        #action space defined as the leverage * [-1,1] if allowing for shorts
        self.leverage = leverage
        self.lower_bound = -1 if self.allow_short == True else 0

        self.action_space = gym.spaces.Box(self.lower_bound*self.leverage, 1*self.leverage,
                                           shape = (len(self.src_data.tickers) + 1,),
                                           dtype = np.float32)

        #get the observation space from data min and max
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf,
                                                shape = (len(tickers), window_size,
                                                         price_history.shape[-1]),
                                                dtype = np.float32)

    def step(self, action):
        return self._step(action)

    #helper function
    def _step(self,action):
        '''
        One step in the env
        Actions correspond to portfolio weights n_ x 1 [w_0, ..., w_n_]
        c_n is the portfolio conversion weights
        '''
        np.testing.assert_almost_equal(action.shape[0],(len(self.sim_data.tickers) + 1,))

        #normalize
        action = np.clip(action,self.lower_bound*self.leverage,1*self.leverage)
        weights = action
        weights /= (weights.sum() + EPS)
        weights[0] += np.clip(1 - weights.sum(), self.lower_bound*self.leverage,1*self.leverage)

        #assert ((action >= self.lower_bound*self.leverage) * (action <= 1*self.leverage)).all()

        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3,
                                       err_msg= 'weights should sum to 1. action = "%s"' % weights)

        observation, done1, ground_truth_obs = self.src_data._step()

        #concatenate obs with ones
        cash_obs = np.ones((observation.shape[0],1))
        observation = np.concatenate((cash_obs, observation),axis=1)

        cash_ground_truth = np.ones((1,1))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis = 1)

        y_1 = observation[-1, :]
        reward, info, done2 = self.sim_data._step(weights, y_1)

        info['market_value'] = np.cumprod([inf['return'] for inf in self.infos + [info]])[-1]
        #add dates
        info['date'] = index_to_date(self.start_idx + self.src_data.idx + self.src_data.step, START_DATE)
        info['steps'] = self.src_data.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim_data.reset()
        observation, ground_truth_obs = self.src_data.reset()
        cash_obs = np.ones((self.window_size, 1))
        observation = np.concatenate((cash_obs, observation), axis = 1)
        cash_ground_truth = np.ones((1,1))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis = 1)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

    def render(self):
        self.plot()

    def plot(self):
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        df_info[['portfolio_value', 'market_value']].plot(fig=plt.gcf(),rot=30)

class StaticEnvironment:
    '''
    Defines the investment universe and Environment for the static case:
    This is the mean-variance and equal weights benchmark that will be compared to the RL application
    '''
    def __init__(self, prices, equity, transaction_cost, mean_model, cov_model, satisfaction_model):
        self.prices = prices
        self.equity = equity
        self.transaction_cost = transaction_cost

        #modelling params
        self.mean_model = mean_model
        self.cov_model = cov_model
        self.satisfaction_model = satisfaction_model

    def get_state(self, t_, d_):
        assert d_ <= t_
        state = self.prices.iloc[d_ : t_ + 1]
        state = state.pct_change().dropna()
        return state

    def get_reward(self, action, t_action, t_reward, alpha = 0.01):
        window = self.prices[t_action : t_reward]
        w = action
        r = window.pct_change().dropna()
        satis = portfolio(w, r, self.mean_model, self.cov_model, self.satisfaction_model)[-1]
        satis = np.array([satis] * len(self.prices.columns))

        return np.dot(r, w), satis

