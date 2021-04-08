import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from mom_models import *
from factor_models import linear_factor_mod
from satis_models import sharpe_ratio

import random
class Agent:
    def __init__(self, n_, is_eval = False, allow_short = False):
        self.input_shape = (n_, n_)
        self.portfolio_size = n_
        self.allow_short = allow_short
        self.is_eval = is_eval
        self.action_size = 3 # {Buy, Hold, Sell}
        self.model = None

    def get_model(self):
        return model.summary()

class Static(Agent):
    def __init__(self, first_moment_estimator, cov_matrix_estimator ,n_, allow_short, **kwargs):
        super().__init__(n_= n_, allow_short=allow_short)
        self.first_moment_estimator = first_moment_estimator
        self.cov_matrix_estimator = cov_matrix_estimator

        self.weight_bound = (0,1)
        if self.allow_short:
            self.weight_bound = (-1,1)

        self.factor_bound = None

    def set_factor_bound(self, bound):
        self.factor_bound = bound

    def set_weight_bound(self, bound: tuple):
        self.weight_bound = bound


    def act(self, returns, optimization_problem, factors = None, **kwargs):
        n_ = int(returns.shape[1])

        #obtain the first and second moment applying the model initialized with the agent
        mu = self.first_moment_estimator(returns)
        sigma = self.cov_matrix_estimator(returns, **kwargs)

        if optimization_problem == 'max_sharpe':
            w_opt_grid = _quadratic_risk_utility(mu, sigma, n_, grid_size = 1000, weight_bound = self.weight_bound, **kwargs)

            w_opt = _search_max_satisfaction(w_opt_grid, returns, sharpe_ratio)

        if optimization_problem == 'minimum_variance':
            w_opt = _minimum_variance(mu, sigma, n_, self.weight_bound)

        if optimization_problem == 'equal_weights':
            w_opt = np.full((n_,1), 1/n_)

        if optimization_problem == 'sr_factor':
            w_opt_grid = _quad_factor_model(mu, returns, factors , grid_size=1000,
                                       weight_bound=self.weight_bound, factor_bound=self.factor_bound)

            w_opt = _search_max_satisfaction(w_opt_grid, returns, sharpe_ratio)

        if optimization_problem == 'equal_risk':
            w_opt = _equal_risk_contribution(mu, sigma)


        return w_opt
        
def _quadratic_risk_utility(mu,sigma, n_, grid_size = 100, weight_bound=None, leverage_bound= None):
    #initialize parameters
    w = cp.Variable(n_)
    gamma = cp.Parameter(nonneg=True)
    returns = mu.T @ w
    risk = cp.quad_form(w, sigma)
    constraints = [cp.sum(w) == 1]

    if weight_bound:
        constraints.append(w >= weight_bound[0])
        constraints.append(w <= weight_bound[1])
    if leverage_bound:
        constraints.append(cp.norm(w, 1) <= leverage_bound)

    problem = cp.Problem(cp.Minimize(gamma*risk - returns), constraints)

    #solve the set of efficient portfolios
    gamma_grid = np.logspace(-3,3, grid_size)

    weights = []
    for i in range(grid_size):
        gamma.value = gamma_grid[i]
        problem.solve()
        weights.append(w.value)

    return weights

def _minimum_variance(mu, sigma, n_, weight_bound = (-1,1), leverage_bound= None):

    w = cp.Variable(n_)
    risk = cp.quad_form(w, sigma)
    constraints = [cp.sum(w) == 1]

    if weight_bound:
        constraints.append(w >= weight_bound[0])
        constraints.append(w <= weight_bound[1])
    if leverage_bound:
        constraints.append(cp.norm(w, 1) <= leverage_bound)

    problem = cp.Problem(cp.Minimize(risk),constraints)
    problem.solve()
    w_opt = w.value

    return w_opt

def _quad_factor_model(mu, returns, factors, grid_size = 1000,
                       weight_bound = (-1,1), leverage_bound = None, factor_bound = None,
                       **kwargs):
    beta, u = linear_factor_mod(returns*100,factors*100, **kwargs)
    E = np.diag(abs(u.sum()))

    w = cp.Variable(E.shape[0])
    gamma = cp.Parameter(nonneg=True)
    f = beta.T@w
    ret = mu.T@w
    risk = cp.quad_form(f, np.cov(beta, rowvar=False)) + cp.quad_form(w,E)

    constraints = [cp.sum(w) == 1]

    if weight_bound:
        constraints.append(w >= weight_bound[0])
        constraints.append(w <= weight_bound[1])
    if leverage_bound:
        constraints.append(cp.norm(w, 1) <= leverage_bound)
    if factor_bound:
        constraints.append(f <= factor_bound)

    problem = cp.Problem(cp.Minimize(gamma * risk - ret), constraints)

    gamma_grid = np.logspace(0,3, grid_size)

    weights = []
    for i in range(grid_size):
        gamma.value = gamma_grid[i]
        problem.solve()
        weights.append(w.value)

    return weights

def _equal_risk_contribution(mu, sigma):

    def portfolio_variance(w, sigma):
        w = np.matrix(w)
        sig_p = w * sigma * w.T
        return sig_p[0,0]

    def risk_contribution(w, sigma):
        # function that calculates asset contribution to total risk
        w = np.matrix(w)
        sigma = np.sqrt(portfolio_variance(w, sigma))
        # Marginal Risk Contribution
        mrc = sigma * w.T
        # Total asset risk contribution
        rc = np.multiply(mrc, w.T) / sigma
        return rc

    def risk_objective(x, params):
        # calculate portfolio risk
        covar = params[0]  # covariance table
        b = params[1]  # risk target in percent of portfolio risk
        sig_p = np.sqrt(portfolio_variance(x, covar)) # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p, b))
        asset_rc = risk_contribution(x, covar)
        J = sum(np.square(asset_rc - risk_target.T))[0, 0]  # sum of squared error
        return J

    def full_investment(x):
        return np.sum(x) - 1.0

    def long_only_constraint(x):
        return x

    b = np.ones((1, mu.shape[0]))
    b = b / mu.shape[0]  # your risk budget percent of total portfolio risk (equal risk)

    cons = ({'type': 'eq', 'fun': full_investment},
            {'type': 'ineq', 'fun': long_only_constraint})

    res = minimize(risk_objective, b, args=[sigma, b], method='SLSQP', constraints=cons, options={'disp': False})

    w_opt = np.array(res.x)
    w_opt = w_opt.reshape(mu.shape[0],)

    return w_opt

def _search_max_satisfaction(w_opt_grid, returns, satisfaction_function):
    satis = np.zeros(len(w_opt_grid))
    for idx, solu in enumerate(w_opt_grid):
        port = (solu * returns).sum(axis=1)
        satis[idx] = satisfaction_function(port)
    max_satis = np.argmax(satis)
    w_opt = w_opt_grid[max_satis]
    return w_opt












