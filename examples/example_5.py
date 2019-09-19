import matplotlib.pyplot as plt
import numpy as np

from RBFN import *

# example V

# generating data
data = np.genfromtxt('Iron_d4.txt')
x = data[:-150, 0]
y = np.subtract(data[:-150, 1], 0.05 * np.random.random(len(x)))

gens = GeneticFitter.GeneticOpt(x, y, items=5, epochs=5, max_iter=50,
                                conv_min=10e-2, hidden_ranges=[15, 145],
                                method=['psed'], plot=True)
y_pred = gens.evolver()
print(gens.__dict__)
# plotting 1D interpolation
plt.plot(x, y, 'b-', label='real')
plt.plot(x, y_pred, 'r--', label='fit')
plt.xlabel("Energy Loss (eV)")
plt.ylabel("Emission Intensity (a.u.)")
plt.legend(loc='best')
plt.title(
	'Using a genetic-optimized RBFN for re-fitting Iron d6 multiplet-spectra')
plt.savefig("example_5.png", dpi=300)
plt.show()
