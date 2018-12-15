#from probo.marketdata import MarketData
#from probo.payoff import *
from probo.engine import *
#from probo.facade import OptionFacade
import numpy as np
#import time

def callPayoff(spot, strike):
    return np.maximum(spot - strike, 0.0)

def putPayoff(spot, strike):
    return np.maximum(strike - spot, 0.0)




## Set up the market data

spot = 100
rate = 0.06
volatility = 0.2
dividend = 0.03
expiry = 1.0
strike = 100
nreps = 1000
steps = 52
convar = 0
erddt = np.exp((rate-dividend)*(expiry/steps))


def AssetPaths(spot, rate, sigma, expiry, div, nreps, nsteps):
    convar = 0
    paths = np.empty((nreps, nsteps + 1))
    h = expiry / nsteps
    paths[:, 0] = spot
    mudt = (rate - div - 0.5 * sigma * sigma) * h
    sigmadt = sigma * np.sqrt(h)

    
    for t in range(1, nsteps + 1):
        z = np.random.normal(size=nreps)
        paths[:, t] = paths[:, t-1] * np.exp(mudt + sigmadt * z) 

    return paths

#def controlVariateAssetPaths(spot, rate, sigma, expiry, div, nreps, nsteps):

#Monte Carlo
paths = AssetPaths(spot, rate, volatility, expiry, dividend, nreps, steps)

callT = callPayoff(paths.max(axis=1), strike)
se = np.std(callT, ddof=1) / np.sqrt(nreps)
prc = callT.mean()
prc *= np.exp(-rate * expiry) 
print("Monte Carlo Call Price: ${0:.2f}".format(prc),"Standard Error:{0:.4f}".format(se))

paths1 = AssetPaths(spot, rate, volatility, expiry, dividend, nreps, steps)
putT = putPayoff(paths1.min(axis=1), strike)
se1 = np.std(putT, ddof=1) / np.sqrt(nreps)
prc1 = putT.mean()
prc1 *= np.exp(-rate * expiry)
print("Monte Carlo Call Price: ${0:.2f}".format(prc1),"Standard Error:{0:.4f}".format(se1))


def antitheticAssetPaths(spot, rate, sigma, expiry, div, nreps, nsteps):
    convar = 0
    paths = np.empty((nreps, nsteps + 1))
    h = expiry / nsteps
    paths[:, 0] = spot
    mudt = (rate - div - 0.5 * sigma * sigma) * h
    sigmadt = sigma * np.sqrt(h)

    
    for t in range(1, nsteps + 1):
        z1 = np.random.normal(size=nreps)
        z2 = -z1
        z = np.concatenate((z1,z2))
        z = np.random.normal(size=nreps)
        paths[:, t] = paths[:, t-1] * np.exp(mudt + sigmadt * z) 

    return paths


paths2 = antitheticAssetPaths(spot, rate, volatility, expiry, dividend, nreps, steps)
callT1 = callPayoff(paths2.max(axis=1), strike)
se2 = np.std(callT1, ddof=1) / np.sqrt(nreps)
prc2 = callT1.mean()
prc2 *= np.exp(-rate * expiry) 
print("Antithetic Call Price: ${0:.2f}".format(prc2), "Standard Error:{0:.4f}".format(se2))

paths3 = antitheticAssetPaths(spot, rate, volatility, expiry, dividend, nreps, steps)
putT1 = putPayoff(paths3.min(axis=1), strike)
se3 = np.std(putT1, ddof=1) / np.sqrt(nreps)
prc3 = putT1.mean()
prc3 *= np.exp(-rate * expiry)
print("Antithetic Put Price: ${0:.2f}".format(prc3), "Standard Error:{0:.4f}".format(se3))

antitheticAssetPaths

def controlVariateAssetPaths(spot, rate, sigma, expiry, div, nreps, nsteps):
    convar = 0
    paths = np.empty((nreps, nsteps + 1))
    h = expiry / nsteps
    paths[:, 0] = spot
    mudt = (rate - div - 0.5 * sigma * sigma) * h
    sigmadt = sigma * np.sqrt(h)
    dt = expiry / nsteps
    erddt = np.exp((rate - dividend)* dt)

    
    for t in range(1, nsteps + 1):
        z = np.random.normal(size=nreps)
        paths[:, t] = paths[:, t-1] * np.exp(mudt + sigmadt * z)
        delta = BlackScholesDelta(spot, t, strike, expiry, volatility, rate, dividend)
        spot_tn = spot * np.exp(mudt + sigmadt * z[t])
        convar = convar + delta * (spot_tn - spot * erddt)
        cashFlow =  spot_tn - convar

    return cashFlow


cashFlow = controlVariateAssetPaths(spot, rate, volatility, expiry, dividend, nreps, steps)
se4 = cashFlow.std() / np.sqrt(nreps)
prc4 = np.exp(-rate * expiry) * cashFlow.mean()
print("Control Variate Call Price: ${0:.2f}".format(prc4), "Standard Error:{0:.4f}".format(se4))

cashFlow = controlVariateAssetPaths(spot, rate, volatility, expiry, dividend, nreps, steps)
se5 = cashFlow.std() / np.sqrt(nreps)
prc5 = np.exp(-rate * expiry) * cashFlow.mean()
print("Control Variate Put Price ${0:.2f}".format(prc5), "Standard Error:{0:.4f}".format(se5))

