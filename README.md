# Generative Stochastic Modeling for Chaotic Dynamical Systems
A Python implementation of the framework proposed in  *"Generative stochastic modeling of strongly
nonlinear flows with non-Gaussian statistics"*
 by H. Arbabi and T. Sapsis (https://arxiv.org/pdf/1908.08941.pdf). 
 
Given time-series data from a measure-preserving chaotic system (a.k.a. stationary stochastic process), this framework generates a system of SDEs with nonlinear observation maps that produce statsitics and power spectra similar to the time series. The framework is based on optimal transport of probabilities, and its summary is shown below.

<img src="https://github.com/arbabiha/StochasticModelingwData/blob/master/thehood/FrameworkSketch.png" width="700">

## What is in here?

**Download the time-series data:** 
Before running the code, go to https://www.dropbox.com/sh/4rr5c8ee0a3szs4/AAClNuOgrDkr-3Ho7GNE8NDUa?dl=0 download the .mat files and place them in "thehood" folder.

**Lorenz96.py** finds a 1D SDE and a 1D nonlinear observation map that reproduces the statistics of a chaotic Lorenz 96 state variable. Runtime is ~5min.

**Cavity.py** finds a 10D SDE and 10D nonlinear map that give the same statistics and PSD as SPOD modal coordinates of cavity flow. Runtime is ~1hr.

**Climate_tails.py** uses the optimal transport model to extrapolate the tails of the distribution (i.e. characterizes the probabilities of extreme events). Runtime is about ~30min.


**Convergence_analysis.py** repeats the modeling for various sample sizes and polynomial orders. For lorenz system it takes 30 min, but for cavity it takes mych longer.


These codes show the use of modules in './thehood/' and reproduce the plots in the paper. The computational bottleneck in modeling is usually computing the inverse of transport maps for samples generated from SDE models. The reported simulation times are based on using 20 parallel threads.

## Dependencies:

[Transportmaps by UQgroup @MIT](https://transportmaps.mit.edu/docs/): This package is used to compute the mapping between data distribution and the invariant measure of SDE system --- here Standard Normal Distribution.

[Numba](https://numba.pydata.org/): Used for fast integration of SDEs.

[joblib](https://joblib.readthedocs.io/en/latest/): Used for parallel computation of inverse function in transportmaps.

[pyswarm](https://pythonhosted.org/pyswarm/): Used for optimization of SDE parameters to match the spectrum of transported data.

Send comments and questions to arbabiha-AT-gmail.com.

