from probo.marketdata import MarketData
from probo.payoff import *
from probo.engine import *
from probo.facade import OptionFacade
import time


## Set up the market data
spot = 100
rate = 0.06
volatility = 0.2
dividend = 0.03
thedata = MarketData(rate, spot, volatility, dividend)
mu = .05
## Set up the option
expiry = 1.0
strike = 100
thecall = ExoticPayoff(expiry, strike, lookbackCallPayoff1)
theput = ExoticPayoff(expiry, strike, lookbackPutPayoff1)
## Set up Naive Monte Carlo?
nreps = 1000
steps = 252

def AssetPaths(spot, mu, sigma, expiry, div, nreps, nsteps):
    paths = np.empty((nreps, nsteps + 1))
    h = expiry / nsteps
    paths[:, 0] = spot
    mudt = (mu - div - 0.5 * sigma * sigma) * h
    sigmadt = sigma * np.sqrt(h)
    
    for t in range(1, nsteps + 1):
        z = np.random.normal(size=nreps)
        paths[:, t] = paths[:, t-1] * np.exp(mudt + sigmadt * z)

    return paths


paths = AssetPaths(spot,mu,volatility,expiry, dividend,nreps,steps)
maxSpot = np.max(paths[1])
minSpot = np.min(paths[1])


#-------------------------------------------------------------------
#MonteCarlo
pricer = NaiveMonteCarloPricer


mcengine = MonteCarloEngine(nreps, steps, pricer)

## Calculate the prices
startTime1 = time.time()
option1 = OptionFacade(thecall, mcengine, thedata)
mcCallPrice, mcCallSe = option1.price()
endTime1 = time.time()
totalTime1 = endTime1 - startTime1



startTime2 = time.time()
option2 = OptionFacade(theput, mcengine, thedata)
mcPutPrice, mcPutSe = option2.price()
endTime2 = time.time()
totalTime2 = endTime2 - startTime2

#-------------------------------------------------------------------

#Control Variate
pricer = ControlVariatePricer


mcengine = MonteCarloEngine(nreps, steps, pricer)

## Calculate the prices
startTime3 = time.time()
option1 = OptionFacade(thecall, mcengine, thedata)
cvCallPrice, cvCallSe = option1.price()
endTime3 = time.time()
totalTime3 = endTime3 - startTime3


startTime4 = time.time()
option2 = OptionFacade(theput, mcengine, thedata)
cvPutPrice, cvPutSe = option2.price()
endTime4 = time.time()
totalTime4 = endTime4 - startTime4

#-------------------------------------------------------------------

#Antithetic
pricer = AntitheticMonteCarloPricer

mcengine = MonteCarloEngine(nreps, steps, pricer)

## Calculate the prices
startTime5 = time.time()
option1 = OptionFacade(thecall, mcengine, thedata)
amcCallPrice, amcCallSe = option1.price()
endTime5 = time.time()
totalTime5 = endTime5 - startTime5


startTime6 = time.time()
option2 = OptionFacade(theput, mcengine, thedata)
amcPutPrice, amcPutSe = option2.price()
endTime6 = time.time()
totalTime6 = endTime6 - startTime6



print("Lookback Call via Naive Monte Carlo:")
print("\t","- Price: ${0:.2f}".format(mcCallPrice))
print("\t","- Time: {0:.4f}".format(totalTime1), "Seconds")
print("\t","- Standard Error: {0:.3f}".format(mcCallSe))

print("\n""Lookback Put via Naive Monte Carlo:")
print("\t","- Price: ${0:.2f}".format(mcPutPrice))
print("\t","- Time: {0:.4f}".format(totalTime2), "Seconds")
print("\t","- Standard Error: {0:.3f}".format(mcPutSe))

print("\n""Lookback Call via Control Variate:")
print("\t","- Price: ${0:.2f}".format(cvCallPrice))
print("\t","- Time: {0:.4f}".format(totalTime3), "Seconds")
print("\t","- Standard Error: {0:.3f}".format(cvCallSe))

print("\n""Lookback Put via Control Variate:")
print("\t","- Price: ${0:.2f}".format(cvPutPrice))
print("\t","- Time: {0:.4f}".format(totalTime4), "Seconds")
print("\t","- Standard Error: {0:.3f}".format(cvPutSe))

print("\n""Lookback Call via Antithetic Monte Carlo:")
print("\t","- Price: ${0:.2f}".format(amcCallPrice))
print("\t","- Time: {0:.4f}".format(totalTime5), "Seconds")
print("\t","- Standard Error: {0:.3f}".format(amcCallSe))

print("\n""Lookback PUt via Antithetic Monte Carlo:")
print("\t","- Price: ${0:.2f}".format(amcPutPrice))
print("\t","- Time: {0:.4f}".format(totalTime6), "Seconds")
print("\t","- Standard Error: {0:.3f}".format(amcPutSe))
