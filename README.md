# Data-driven Stochastic Modeling
The framework proposed in  *"Data-driven modeling of strongly nonlinear chaotic systems with non-Gaussian statistics"*
 by H. Arbabi and T. Sapsis.
 
Given time series data on a measure-preserving chaotic system, this framework generates a system of SDEs with nonlinear observation maps which produce the similar statsitics and spectra as the time series. Here is the schematic representation:

<img src="https://github.com/arbabiha/StochasticModelingwData/blob/master/thehood/FrameworkSketch.png" width="700">

#### files in the root folder:

#### Lorenz 96: 
Finds a 1D SDE and a 1D nonlinear observation map that reproduces the statistics of a chaotic Lorenz 96 state variable

#### Cavity: 
Finds a 10D SDE and 10D nonlinear map that give the same stat as SPOD modal coordinates. Before running this download the cavity time series data from ... and put in "thehood" folder.

#### Climate: 
Given chaotic climate time series, it uses the transport to extrapolate the tails of the distribution.


## Dependencies:
