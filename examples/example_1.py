import matplotlib.pyplot as plt
import numpy as np

from RBFN import *

# example I

# generating data
x = np.linspace(-5, 5, 250)
y = np.sin(x) + np.cos(x) ** 2 + np.random.uniform(-0.1, 0.1, 250)

gens = GeneticFitter.GeneticOpt(x, y, items=10, epochs=4, max_iter=10, conv_min=10e-4, hidden_ranges=[5, 145],
                                method=['gaus'],
                                plot=True)
y_pred = gens.evolver()
print(gens.__dict__)
# plotting 1D interpolation
plt.plot(x, y, 'b-', label='real')
plt.plot(x, y_pred, 'r-', label='fit')
plt.legend(loc='best')
plt.title('Interpolation using a genetic-optimized RBFN')
plt.savefig("example_1.png",dpi=300)
plt.show()
