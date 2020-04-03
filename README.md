# Data-driven Stochastic Modeling
A Python implementation of the framework proposed in  *"Data-driven modeling of strongly nonlinear chaotic systems with non-Gaussian statistics"*
 by H. Arbabi and T. Sapsis.
 
Given time series data from a measure-preserving chaotic system (a.k.a. stationary stochastic process), this framework generates a system of SDEs with nonlinear observation maps that produce statsitics and power spectra similar to the time series. The framework is based on optimal transport of probabilities, and its summary is shown in the figure below.

<img src="https://github.com/arbabiha/StochasticModelingwData/blob/master/thehood/FrameworkSketch.png" width="700">

## files in the root folder:

#### Data: 
Before running the code, go to https://www.dropbox.com/sh/yri4i4r90glh8q2/AADbfXQ0FFGstQbXAM1sdvw6a?dl=0 download the .mat files and place them in "thehood" folder.

#### Lorenz 96: 
Finds a 1D SDE and a 1D nonlinear observation map that reproduces the statistics of a chaotic Lorenz 96 state variable. Runtime is ~5min.

#### Cavity: 
Finds a 10D SDE and 10D nonlinear map that give the same statistics and PSD as SPOD modal coordinates of cavity flow. Runtime is ~1hr.

#### Climate: 
It uses the optimal transport model to extrapolate the tails of the distribution (i.e. characterizes the probabilities of extreme events). Runtime is about ~30min.

These codes show the use of modules in './thehood/' and reproduce the plots in the paper. The computational bottleneck in all three is computing the inverse of transport maps for samples generated from SDE models. The reported simulation times are based on using 20 parallel threads.

## Dependencies:

[Transportmaps by UQgroup @MIT](https://transportmaps.mit.edu/docs/): This package is used to compute the mapping between data distribution and the invariant measure of SDE system --- here Standard Normal Distribution.

[Numba](https://numba.pydata.org/): Used for fast integration of SDEs.

[joblib](https://joblib.readthedocs.io/en/latest/): Used for parallel computation of inverse function in transportmaps.

[pyswarm](https://pythonhosted.org/pyswarm/): Used for optimization of SDE parameters to match the spectrum of transported data.

Send comments and questions to arbabiha-AT-gmail.com.

