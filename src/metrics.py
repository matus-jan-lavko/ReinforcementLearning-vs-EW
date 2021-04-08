import numpy as np
import pandas as pd

def annualizedRet(r, num_periods):
    '''

    @param r: series, returns
    @param num_periods: scalar, number of periods in return series
    @return: scalar, annualized return
    '''

    comp = (1 + r).prod()
    nPer = r.shape[0]
    return comp ** (num_periods / nPer) - 1

def annualizedVol(r, num_periods, downside = False):
    '''

    @param r: series, returns
    @param num_periods: scalar, number of periods in return series
    @param downside: bool, for downside std
    @return: scalar, annualized volatility
    '''

    if downside:
        semistd = r[r < 0].std()
        return semistd * (num_periods ** 0.5)
    else:
        return r.std() * (num_periods ** 0.5)


def drawdown(r: pd.Series):
    '''

    @param r: series, returns
    @return: dictionary: 'hwm':high watermark, 'drawdowns': drawdown periods
    '''

    index = 1000 * (1 + r).cumprod()
    highwatermark = index.cummax()
    drawdowns = (index - highwatermark) / highwatermark
    return pd.DataFrame(dict(hwm = highwatermark,
                             drawdowns=drawdowns))

def skewness(r):
    '''

    @param r: series, returns
    @return: scalar, third moment
    '''
    centerMoment = r - r.mean()
    sigR = r.std(ddof=0)
    exp = (centerMoment ** 3).mean()
    return exp / sigR ** 3

def kurtosis(r):
    '''

    @param r: series, returns
    @return: scalar, fourth moment
    '''

    centerMoment = r - r.mean()
    sigR = r.std(ddof=0)
    exp = (centerMoment ** 4).mean()
    return exp / sigR ** 4

def varGaussian(r, level=5, modified=False):
    '''

    @param r: series, returns
    @param level: scalar, significance level
    @param modified: bool, taylor expansion and approximation of the VAR
    @return: scalar, percentage of portfolio Value at Risk
    '''
    from scipy.stats import norm
    z = norm.ppf(level / 100)

    if modified is True:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return - (r.mean() + z * r.std(ddof=0))

def sharpeRatio(r, rf, num_periods):
    '''

    @param r: series, returns
    @param rf: scalar or series, of risk-free rate proxy
    @param num_periods: scalar, number of periods
    @return: scalar, risk-adjusted return
    '''
    # convert the annual riskfree to per period
    rf = (1 + rf) ** (1 / num_periods) - 1
    excessRets = r - rf
    annExcessRets = annualizedRet(excessRets, num_periods)
    annVol = annualizedVol(r, num_periods)
    return annExcessRets / annVol

def sortinoRatio(r,rf, num_periods):
    '''

    @param r: series, returns
    @param rf: scalar or series, of risk-free rate proxy
    @param num_periods: scalar, number of periods
    @return: scalar, risk-adjusted return
    '''

    rf = (1 + rf) ** (1 / num_periods) - 1
    excessRets = r - rf
    annExcessRets = annualizedRet(excessRets, num_periods)
    anndownsideVol = annualizedVol(r, num_periods, downside=True)
    return annExcessRets / anndownsideVol

def summary_stats(r, riskFree=0, periodsInYear=252):
    '''

    @param r: series, return
    @param riskFree: scalar or series, of risk-free rate proxy
    @param num_periods: scalar, number of periods
    @param title: string, title of the returned df
    @return: DataFrame of summary statistics
    '''

    if not isinstance(r,pd.DataFrame):
        r = pd.DataFrame(r)

    annR = r.aggregate(annualizedRet, num_periods= periodsInYear)
    annVol = r.aggregate(annualizedVol, num_periods= periodsInYear)
    dd = r.aggregate(lambda r: drawdown(r).drawdowns.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    modVar = r.aggregate(varGaussian, level=5, modified=True)
    sharpe = r.aggregate(sharpeRatio, rf=riskFree, num_periods = periodsInYear)
    sortino = r.aggregate(sortinoRatio, rf = riskFree, num_periods = periodsInYear)

    stats = pd.DataFrame({
        'Annualized Returns': annR*100,
        'Annualized Volatility':  annVol*100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown':  dd*100,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish Fisher adj. VAR 5%': modVar*100,
    })

    #formatting
    stats['Annualized Returns'] = stats['Annualized Returns'].map('{:,.2f}%'.format)
    stats['Annualized Volatility'] = stats['Annualized Volatility'].map('{:,.2f}%'.format)
    stats['Sharpe Ratio'] = stats['Sharpe Ratio'].map('{:,.2f}'.format)
    stats['Sortino Ratio'] = stats['Sortino Ratio'].map('{:,.2f}'.format)
    stats['Max Drawdown'] = stats['Max Drawdown'].map('{:,.2f}%'.format)
    stats['Skewness'] = stats['Skewness'].map('{:,.2f}'.format)
    stats['Kurtosis'] = stats['Kurtosis'].map('{:,.2f}'.format)
    stats['Cornish Fisher adj. VAR 5%'] = stats['Cornish Fisher adj. VAR 5%'].map('{:,.2f}%'.format)

    return stats.T