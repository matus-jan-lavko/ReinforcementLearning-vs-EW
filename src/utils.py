import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import time
import os
import imageio
from tqdm import tqdm
import sys


import statsmodels.api as sm
from scipy.stats import shapiro

from environment import StaticEnvironment

def date_to_index(date_string, start_date):
    return (dt.datetime.strptime(date_string, '%Y-%m-%d') - start_date).days

def index_to_date(index, start_date):
    return(start_date + dt.timedelta(index)).strftime('%Y-%m-%d')


def portfolio(w :np.array, r: np.array, mean_model, cov_model, satisfaction_model, annualize = False):
    '''

    :param w: n x 1 portfolio weights
    :param r: t x n portfolio returns
    :param mean_model: function for modelling the expected value of the portfolio
    :param cov_model: function for modelling the covariance matrix
    :param satisfaction: satisfaction function
    :return:
    '''

    mu_hat = mean_model(r)
    sig_hat = cov_model(r)


    if annualize:
        mu_hat *= 252
        sig_hat *= 252

    r_p = np.sum(mu_hat * w)
    sig_p = np.sqrt(np.dot(w.T, np.dot(sig_hat, w)))

    #satisfaction measure
    satis = satisfaction_model(r_p, sig_p)
    return np.array([r_p, sig_p, satis])


def test_static_agent(env, agent, optimization_problem,
                      fitting_period, rebalancing_period,
                      factors=None, **kwargs):
    returns = []
    actions = []
    counter = 0
    tic = time.perf_counter()

    # factor indexing
    if optimization_problem == 'sr_factor':
        factor_env = StaticEnvironment(factors.loc[env.prices.index[0]:], **kwargs)

    for trade in range(fitting_period, len(env.prices), rebalancing_period):
        #         print(trade, counter*rebalancing_period)

        s_t = env.get_state(trade, counter * rebalancing_period)
        #         print(s_t.shape)

        if optimization_problem == 'sr_factor':

            s_t_factor = env.get_state(trade, counter * rebalancing_period)
            a_t = agent.act(s_t, optimization_problem, factors=s_t_factor)

        else:
            a_t = agent.act(s_t, optimization_problem, **kwargs)

        actions.append(a_t)
        s_t_trade = s_t.iloc[-rebalancing_period:, :]

        #transaction costs
        if len(actions) > 1:
            a_delta = actions[len(actions) - 1] - actions[len(actions) - 2]

            r_t = np.dot(s_t_trade, a_t) - np.dot(s_t_trade * env.transaction_cost, a_delta)

        else:
            r_t = np.dot(s_t_trade, a_t)

        returns.append(r_t)

        counter += 1

    returns = np.vstack(returns).flatten()
    toc = time.perf_counter()
    print(f"Tested {optimization_problem} in {toc - tic:0.4f} seconds")
    return returns


def load_data_long(db_path):
    db_long_prices = pd.read_csv(db_path + '00_db_long__PX_LAST.csv', index_col=0, parse_dates=True)
    db_long_prices = db_long_prices.loc['2015':]
    db_long_RL = db_long_prices.loc[:, ~db_long_prices.iloc[0].isna()].fillna(method='ffill')
    return db_long_RL

def plot_training_result(rl_history, benchmark_history, n_, actions_to_plot, column_names):

    rl_result = np.array(rl_history).cumsum()
    benchmark_result = np.array(benchmark_history).cumsum()

    fig = plt.figure(figsize=(12,6))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=4)
    bottom = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=4)

    #returns
    top.plot(rl_result, color='black', ls = '-')
    top.plot(benchmark_result, color = 'grey', ls = '--')

    #weights
    for a in actions_to_plot:
        plt.bar(np.arange(n_), a, color = 'goldenrod', alpha = 0.25)
        plt.xticks(np.arange(n_), column_names, rotation = 'vertical')
    plt_show()

def plot_histograms(ew, subset):
    sns.set_palette('bright')
    fig, ax = plt.subplots(figsize=(20, 15))
    for i, column in enumerate(subset.columns, 1):
        plt.subplot(3, 3, i)
        to_plot = pd.concat([ew, subset[column]], axis=1)
        sns.histplot(to_plot, kde=True, multiple='stack', alpha=0.5)
        plt.xlim(-.13,.13)


def create_weights_gif(weights, model_name, saving_path, **plot_kwargs):
    '''

    @param weights: array of weights
    @param model_name: name of the model, string
    @param saving_path: path to save, string
    @param plot_kwargs: list of kwargs to unpack for plot
    @return: None
    '''

    tic = time.perf_counter()
    n_frames = 5
    x = weights.columns.to_list()

    # obtain lists of weights for each day
    y_lists = []
    for _, row in weights.iterrows():
        rw = row.to_list()
        y_lists.append(rw)

    # iterate over each row
    filenames = []
    y_cache = []

    with tqdm(total=round(len(y_lists) / 20, 0), file=sys.stdout) as pbar:
        for index in np.arange(0, len(y_lists) - 1, step=20):
            y = y_lists[index]
            y1 = y_lists[index + 1]

            # distance to next pos
            y_path = np.array(y1) - np.array(y)

            # obtain image for each frame
            for i in np.arange(0, n_frames + 1):
                y_temp = (y + (y_path / n_frames) * i)

                y_cache.append(y_temp)

                # plot
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.barh(x, y_temp, color='goldenrod', **plot_kwargs)

                # cache and plot dissipating weights
                if len(y_cache) > 0:
                    for idx, cache in enumerate(y_cache):
                        plt.barh(x, cache, color='goldenrod', alpha=0.4 - 0.05 * idx)

                plt.xlim(0, 0.07)

                # if cache is full first in last out
                if len(y_cache) == 8:
                    y_cache.pop(0)

                # build a filename
                filename = os.path.join(saving_path, f'gif/frame_{index}_{i}.png')
                filenames.append(filename)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_title(f'{model_name} test trading day: #{index}')
                ax.set_xlabel('weight')

                # last frame needs to stay longer
                if (i == n_frames):
                    for i in range(2):
                        filenames.append(filename)

                # save images
                plt.savefig(filename, dpi=96)
                plt.close()

            pbar.update(1)

    print('Charts saved \n')

    print('Creating gif\n')

    # create the gif
    with imageio.get_writer(os.path.join(saving_path, f'{model_name}_weights.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    toc = time.perf_counter()
    print(f'Gif produced in {(toc - tic) / 60 :0.4f} minutes')

    # print('Removing Images\n')
    # # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)
    print('DONE')


def normality_test(ew, subset):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    ax = axes.flatten()
    subset = pd.concat([ew, subset], axis=1)
    subset = subset.iloc[1:, ]
    for i, column in enumerate(subset.columns):
        _, p_value = shapiro(subset[column])

        sm.qqplot(subset[column], line='q', ax=ax[i])
        ax[i].set_title(column + ': Shapiro-Wilk p-value:' + str(round(p_value, 4)))


def plt_show():
    '''Text-blocking version of plt.show()
    Use this instead of plt.show()'''
    plt.draw()
    plt.pause(0.001)
    input("Press enter to continue...")
    plt.close()

    # convert List of weights to DataFrame
    weights = pd.DataFrame(weightsRebal, index=r.iloc[estimation_window:].index, columns=r.columns)
    returns = (weights * r).sum(axis="columns", min_count=1)  # mincount is to generate NAs if all inputs are NAs
    return returns


def read_ohlcv(db_name, db_path):
    c = pd.read_csv(db_path + db_name + '__PX_LAST.csv', parse_dates=True, index_col=0)
    c = c.loc[:, ~c.iloc[0].isna()]

    #drop names that were not traded
    c = c.loc[:, ~c.iloc[-1].isna()]
    ticks = c.columns

    o = pd.read_csv(db_path + db_name + '__PX_OPEN.csv', parse_dates=True, index_col=0).reindex(ticks, axis = 1)
    h = pd.read_csv(db_path + db_name + '__PX_HIGH.csv', parse_dates=True, index_col=0).reindex(ticks, axis = 1)
    l = pd.read_csv(db_path + db_name + '__PX_LOW.csv', parse_dates=True, index_col=0).reindex(ticks, axis = 1)

    volu = pd.read_csv(db_path + db_name + '__PX_VOLUME.csv', parse_dates=True, index_col=0).reindex(ticks, axis = 1)

    dc = {'o': o, 'h' : h, 'l' : l, 'c' : c, 'volu' : volu}
    return dc

def preprocess_custom_data(o, h, l, c, volu, start_date):
    '''
    input custom data in wide format and get the data in the long format for FinRL
    '''
    o = o.loc[start_date:].melt(ignore_index=False, value_name='open', var_name='tic')
    h = h.loc[start_date:].melt(ignore_index=False, value_name='high', var_name='tic')
    l = l.loc[start_date:].melt(ignore_index=False, value_name='low', var_name='tic')
    c = c.loc[start_date:].melt(ignore_index=False, value_name='close', var_name='tic')
    volu = volu.loc[start_date:].melt(ignore_index=False, value_name='volume', var_name='tic')

    df = pd.concat([o, h['high']], axis = 1).fillna(method='ffill')
    df = pd.concat([df,l['low']], axis = 1).fillna(method='ffill')
    df = pd.concat([df,c['close']], axis = 1).fillna(method='ffill')
    df = pd.concat([df,volu['volume']], axis = 1).fillna(method='ffill')

    df = df.reset_index().rename(columns={'index': 'date'})
    df['day'] = df.groupby(['date']).ngroup() + 1  # get id for each day

    return df.sort_values(by='date')

def add_cov_matrix_states(df, lookback):
    # add covariance matrix as states
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    # look back is one year
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        #think about implementing shrinkage etc
        sigma = return_lookback.cov().values
        cov_list.append(sigma)

    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    return df

def backtestWs(prices, weighting, estimation_window=60, rebalancing_window=22,**kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """

    try: r = prices.pct_change()
    except: print('You need to input a price matrix')

    n_periods = r.shape[0]
    # return windows
    windows = [(start, start + estimation_window) for start in range(n_periods - estimation_window)]

    if hasattr(weighting,'__call__'):
        if estimation_window > 0:
            weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
        else:
            weights = [weighting(r.iloc[win[0]], **kwargs) for win in windows]
    else:
        weights = [weighting.iloc[win[0], :] for win in windows]

    weights = np.array(weights)

    # rebalancing
    weightsRebal = []
    counter = 0
    # rebalancing window computation
    for i in range(n_periods - estimation_window):
        if counter == 0:
            currW = weights[0]
        if counter % rebalancing_window == 0:
            currW = weights[i]
        counter += 1
        weightsRebal.append(currW)

    # convert List of weights to DataFrame
    weights = pd.DataFrame(weightsRebal, index=r.iloc[estimation_window:].index, columns=r.columns)
    returns = (weights * r).sum(axis="columns", min_count=1)  # mincount is to generate NAs if all inputs are NAs

    return returns

def weightsEw(r):
    '''returns weights of the EW portfoio based on r'''
    r = r.notna()
    weight = [1/r.shape[0] for _ in range(r.shape[0])]
    return weight