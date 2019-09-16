import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from RBFN import *



# generating data
x, y = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
z = (np.cos(np.sqrt((x - 2.) ** 2 + (y - 1) ** 2)) - np.sin(np.sqrt((x + 2.) ** 2 + (y + 4) ** 2))) / 2.

# fitting RBF-Network with data
features = np.asarray(list(zip(x.flatten(), y.flatten())))



model = RBFNetwork.RBFN(hidden_shape=100)
model.fit(features, z.flatten())
predictions, mse, params, conv  = model.scf(X=features,y=z.flatten(),max_iter=20)

# plotting 2D interpolation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
fig.suptitle('RBF-Network 2D interpolation with a SCF-Cycle', fontsize=20)

ax1.set_title('real', fontsize=20)
ax1.contourf(x, y, z,cmap=cm.Spectral)
ax2.set_title('real', fontsize=20)
ax2.contourf(x, y, predictions.reshape(50, 50),cmap=cm.Spectral)
plt.show()
