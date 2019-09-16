# RBF Network Fitting

**RBF Network Fitting** is a in Python developed fitting routine, which is using the [Radial-Basis-Function-Network for solving](https://en.wikipedia.org/wiki/Radial_basis_function_network) the 1D- and 2D-minimization problem. During the *Self-Consistent-Field-Optimization* of the RBF-Network, the `mean-squared-error` will be evaluated during each cycle, and a difference- and gradient-correction will be applied to the input-parameter of the Fitting-Model. As Fitting-Models can be choosen: 
 * [Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
 * [Cauchy/Lorentzian Distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
 * [Pseudo-Voigt Profile](https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_Approximation)

In order to optimize the *Hyperparameter-Finding* for the number of layers and the kind of choosen models, a [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) can be optionally used.


## Examples

* Detecting peaks of an oscillating function

Example - I             |  Example - II
:-------------------------:|:-------------------------:
<img src="https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/1D_example.png" width="324" height="324">|<img src="https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/1D_example_osci.png" width="324" height="324">


* Fitting of experimental data
<img src="https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/1D_example_experimentlike.png" width="450" height="450">

* Following patterns of 3D-Functions

Example - III             |  Example - IV
:-------------------------:|:-------------------------:
![3D-I](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/3D_example_I.png)|![3D-II](https://github.com/Anselmoo/RBF_NetworkFitting/blob/master/docu/3D_example_II.png)

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
