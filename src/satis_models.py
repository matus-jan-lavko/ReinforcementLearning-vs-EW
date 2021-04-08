import numpy as np

def sharpe_ratio(port, r_f = 0, annualize = True):
    sr = (port.mean()/port.std()-r_f)
    if annualize:
        return sr * 252 ** 0.5
    else:
        return sr



