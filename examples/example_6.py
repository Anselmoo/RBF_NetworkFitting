import matplotlib.pyplot as plt
import numpy as np

from RBFN import *

# example II

# generating data
x = np.linspace(-25, 25, 500)
y = np.sin(x)  * np.cos(x) ** 2  + np.random.uniform(-0.1, 0.1, 500)

# fitting RBF-Network with data
model = RBFNetwork.RBFN(hidden_shape=150)

# model.fit(x, y)

y_pred, mse, params, conv = model.scf(x, y, max_iter=15)

# plotting 1D interpolation
plt.plot(x, y, 'b-', label='real')
plt.plot(x, y_pred, 'r--', label='fit')
plt.legend(loc='upper right')
plt.title('Interpolation using a RBFN')
plt.savefig("example_6.png",dpi=300)
plt.show()
