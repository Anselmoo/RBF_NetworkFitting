[![Build Status](https://travis-ci.com/Anselmoo/RBF_NetworkFitting.svg?branch=master)](https://travis-ci.com/Anselmoo/RBF_NetworkFitting)
[![CodeFactor](https://www.codefactor.io/repository/github/anselmoo/rbf_networkfitting/badge)](https://www.codefactor.io/repository/github/anselmoo/rbf_networkfitting)
[![codebeat badge](https://codebeat.co/badges/9ef976e1-f0f3-4d03-a9d0-23d71a44584b)](https://codebeat.co/projects/github-com-anselmoo-rbf_networkfitting-master)
[![Mergify Status](https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/Anselmoo/RBF_NetworkFitting&style=flat)](https://github.com/Anselmoo/RBF_NetworkFitting/commits/master)
[![DOI](https://zenodo.org/badge/208275828.svg)](https://zenodo.org/badge/latestdoi/208275828)
[![GitHub](https://img.shields.io/github/license/Anselmoo/RBF_NetworkFitting)](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/LICENSE)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/Anselmoo/RBF_NetworkFitting)](https://github.com/Anselmoo/RBF_NetworkFitting/releases)
# RBF Network Fitting

**RBF Network Fitting** is an in Python developed fitting routine, which is using the [Radial-Basis-Function-Network for solving](https://en.wikipedia.org/wiki/Radial_basis_function_network) the 1D- and 2D-minimization problem. During the *Self-Consistent-Field-Optimization* of the RBF-Network, the `mean-squared-error` will be evaluated for each cycle, and a *difference- and gradient-correction* will be applied to the input-parameter of the Fitting-Model. As Fitting-Models can be choosen: 
 * [Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
 * [Cauchy/Lorentzian Distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
 * [Pseudo-Voigt Profile](https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_Approximation)

In order to optimize the *Hyperparameter-Finding* for the number of layers and the kind of choosen models, a [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) can be optionally used. The combination of both *Radial-Basis-Function-Network* and *Genetic Algorithm* allows using **RBF Network Fitting** as a real `black-box-method` in the absence of empirical parameters.


## Examples

* Detecting peaks of an oscillating function

Example - I             |  Example - II
:-------------------------:|:-------------------------:
![osci_1](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/example_2.png)|![osci_2](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/example_6.png)


* Fitting of experimental data

Example - III             |  
:-------------------------:|
![d6_example](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/example_5.png)|

* Following patterns of 3D-Functions

Example - IV             |  Example - V
:-------------------------:|:-------------------------:
![3D-I](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/example_4.png)|![3D-II](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/example_7.png)

**RBF Network Fitting** requires:
  * [numpy](https://github.com/numpy/numpy)
  * [matplotlib](https://github.com/matplotlib/matplotlib)
  
 
 Installing and Running:
```python 
python setup.py install
# as command line application 
python -m RBFN 
# as library
from RBFN import GeneticFitter
from RBFN import RBFNetwork
from RBFN import PlotResults
```

## Further Readings:
```
Genetic Algorithms and Machine Learning for Programmers: Create AI Models and Evolve Solutions
Frances Buontempo
Pragmatic Bookshelf, 2019
```

```
Genetic Algorithms with Python
Clinton Sheppard
Clinton Sheppard, 2018
```
[https://github.com/handcraftsman/GeneticAlgorithmsWithPython/blob/master/ch08/genetic.py](https://github.com/handcraftsman/GeneticAlgorithmsWithPython/blob/master/ch08/genetic.py)
[https://en.wikipedia.org/wiki/Radial_basis_function_network](https://en.wikipedia.org/wiki/Radial_basis_function_network)    
    

