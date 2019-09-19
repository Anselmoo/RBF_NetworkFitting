import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from RBFN import *

# example IV

# generating data
x, y = np.meshgrid(np.linspace(0, 15, 50), np.linspace(0, 15, 50))
z = 2 * (np.cos(np.sqrt((x) ** 2 + (y) ** 2)) ** 2 - np.sin(
    np.sqrt((x) ** 2 + (y) ** 2))) / 2.

# fitting RBF-Network with data
features = np.asarray(list(zip(x.flatten(), y.flatten())))

model = RBFNetwork.RBFN(hidden_shape=200)
model.fit(features, z.flatten())
predictions, mse, params, conv = model.scf(X=features, y=z.flatten(),
                                           max_iter=20, conv_min=10e-4)

# plotting 3D


fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

fig.suptitle('RBF-Network 2D interpolation with a SCF-Cycle', fontsize=20)

ax1.set_title('real', fontsize=20)
ax1.plot_surface(x, y, z, cmap=cm.Spectral)
ax2.set_title('fit', fontsize=20)
ax2.plot_surface(x, y, predictions.reshape(50, 50), cmap=cm.Spectral)
plt.savefig("example_7.png", dpi=300)
plt.show()
