import numpy as np

def mean_return_historic(r):
    '''
    Computes mean historical expected returns

    r: t x n pd.DataFrame of returns

    out: 1 x n np.array of expected returns
    '''
    comp = (1 + r).prod()
    nPer = r.shape[0]
    annr = comp ** (252 / nPer) - 1
    return annr.values

def ema_return_historic(r, window = 22):
    '''
    Computes exponentially weighted historical returns

    r: t x n pd.DataFrame of returns
    window: window for calculating exponential moving average

    out: 1 x n np.array of expected returns
    '''
    ema = (1 + r.ewm(span = window).mean().iloc[-1])
    annr = ema ** 252 - 1
    return annr.values


def sample_cov(r):
    '''
    Sample empirical covariance estimator

    r: t x n pd.DataFrame of returns

    out: n x n np.array of covariances
    '''
    return np.cov(r,rowvar=False)*252

def elton_gruber_cov(r):
    '''
    Constant correlation model of Elton and Gruber

    r: t x n pd.DataFrame of returns

    out: n x n np.array of covariances
    '''
    rho = r.corr()
    n_ = rho.shape[0]

    rho_bar = (rho.sum()-n_)/(n_*(n_-1))
    ccor = np.full_like(rho, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sig = r.std()
    ccov = ccor * np.outer(sig, sig)
    return ccov

def shrinkage_cov(r, delta = 0.5, prior_model = elton_gruber_cov):
    '''
    Shrinks the sample covariance towards a specified model such as Elton/Gruber

    r: n x 1 returns
    delta: shrinkage parameter [0,1] #0.5 is standard
    prior_model: prior_model function that we shrink towards

    out: n x n shrunk covariance matrix
    '''

    prior = prior_model(r, **kwargs)
    sig_hat = sample_cov(r)

    #https://jpm.pm-research.com/content/30/4/110
    honey = delta*prior + (1-delta)*sig_hat
    return honey





