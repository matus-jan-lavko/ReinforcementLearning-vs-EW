import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LinearRegression

def linear_factor_mod(y, x, p = None, regularize = None, return_alpha = False):
    t_, n_ = y.shape
    #set uniform weights if no kernel
    if p is None:
        p = np.ones(t_) / t_

    m_y = p @ y
    m_x = p @ x

    y_p = ((y - m_y).T * np.sqrt(p)).T
    x_p = ((x - m_x).T * np.sqrt(p)).T

    #fit the model
    if regularize == 'L1':
        mod = Lasso(alpha = 0.01/(2.*t_), fit_intercept = True)
    if regularize == 'L2':
        mod = Ridge(alpha = 0.01/(2.*t_), fit_intercept = True)
    if regularize == 'net':
        mod = ElasticNet(alpha = 0.01/(2.*t_), fit_intercept = True)
    else:
        mod = LinearRegression()

    mod.fit(x_p, y_p)

    #retrieve coefficients and residuals
    beta = mod.coef_

    alpha = mod.intercept_
    u = np.subtract(y - alpha, x @ np.atleast_2d(beta.T))

    if not return_alpha:
        return beta, u

    return alpha, beta, u

