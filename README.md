# RBF Network Fitting

**RBF Network Fitting** is a in Python developed fitting routine, which is using the [Radial-Basis-Function-Network for solving](https://en.wikipedia.org/wiki/Radial_basis_function_network) the 1D- and 2D-minimization problem. In order to optimize the Hyperparameter-Finding, a [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) can be optionally used.


|# |	MSE	|	Difference |	Gradient
---|------|------------|----------
1	| 0.01811 |	-0.00334106	| 0.00001545
2	| 0.01785 |	-0.00330281	| 0.00001573
3	| 0.01760 |	-0.00326501	| 0.00001601
4	| 0.01736 |	-0.00322764	| 0.00001628
5	| 0.01712 |	-0.00319073	| 0.00001655
6	| 0.01688 |	-0.00315427	| 0.00001681
7	| 0.01665 |	-0.00311826	| 0.00001707
8	| 0.01643 |	-0.00308272	| 0.00001732
9	| 0.01621 |	-0.00304765	| 0.00001756



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
