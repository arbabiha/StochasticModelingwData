# Data-driven Stochastic Modeling
A Python implementation of the framework proposed in  *"Data-driven modeling of strongly nonlinear chaotic systems with non-Gaussian statistics"*
 by H. Arbabi and T. Sapsis.
 
Given time series data on a measure-preserving chaotic system, this framework generates a system of SDEs with nonlinear observation maps which produce statsitics and spectra similar to the time series. Here is a schematic representation:

<img src="https://github.com/arbabiha/StochasticModelingwData/blob/master/thehood/FrameworkSketch.png" width="700">

## files in the root folder:

#### Data: 
Before running the code go to https://www.dropbox.com/sh/yri4i4r90glh8q2/AADbfXQ0FFGstQbXAM1sdvw6a?dl=0 download the .mat files and place them in "thehood" folder.

#### Lorenz 96: 
Finds a 1D SDE and a 1D nonlinear observation map that reproduces the statistics of a chaotic Lorenz 96 state variable. Runtime ~ 5 min.

#### Cavity: 
Finds a 10D SDE and 10D nonlinear map that give the same stat as SPOD modal coordinates. Runtime ~1hr.

#### Climate: 
Given chaotic climate time series, it uses the transport to extrapolate the tails of the distribution. Runtime ~ 0.5 hr.

The bottleneck of above codes is computing the inverse of transport maps for SDE models. The reported simulation times are based on using 20 parallel threads.

## Dependencies:

[Transportmaps by UQgroup @MIT](https://transportmaps.mit.edu/docs/): This package is used to compute the mapping between data distribution and the invariant measure of SDE system --- here Standard Normal Distribution.

[Numba](https://numba.pydata.org/): Used for fast integration of SDEs.

[joblib](https://joblib.readthedocs.io/en/latest/): Used for parallel computation of inverse function in transportmaps.

[pyswarm](https://pythonhosted.org/pyswarm/): Used for optimization of SDE parameters to match the spectrum of transported data.

Send comments and questions to arbabiha-AT-gmail.com.

