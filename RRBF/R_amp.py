"""
File: RBFN.py
Author: Octavio Arriaga
Email: arriaga.camargo@email.com
Github: https://github.com/oarriaga
Description: Minimal implementation of a radial basis function network
"""

import numpy as np


class RBFN(object):

	def __init__(self, hidden_shape, sigma=1.0):
		""" radial basis function network
		# Arguments
			input_shape: dimension of the input data
			e.g. scalar functions have should have input_dimension = 1
			hidden_shape: the number
			hidden_shape: number of hidden radial basis functions,
			also, number of centers.
		"""
		self.hidden_shape = hidden_shape
		self.sigma = sigma
		self.centers = None
		self.ampls = None
		self.centers_0 = None
		self.ampls_0 = None
		self.weights = None


	def _kernel_function(self, center, data_point, ampl=1.):
		return ampl * np.exp(-self.sigma * np.linalg.norm(center - data_point) ** 2)

	def _calculate_interpolation_matrix(self, X):
		""" Calculates interpolation matrix using a kernel_function
		# Arguments
			X: Training data
		# Input shape
			(num_data_samples, input_shape)
		# Returns
			G: Interpolation matrix
		"""
		G = np.zeros((len(X), self.hidden_shape))
		for data_point_arg, data_point in enumerate(X):
			for center_arg, center in enumerate(self.centers):
				G[data_point_arg, center_arg] = self._kernel_function(center, data_point, ampl=self.ampls[center_arg])
		return G

	def _select_centers(self, X):
		random_args = np.random.choice(len(X), self.hidden_shape)
		centers = X[random_args]
		return centers

	def _select_ampl(self, X):

		random_args = np.random.choice(len(X), self.hidden_shape)
		ampls = X[random_args]
		return ampls

	def fit(self, X, Y,init=True):
		""" Fits weights using linear regression
		# Arguments
			X: training samples
			Y: targets
		# Input shape
			X: (num_data_samples, input_shape)
			Y: (num_data_samples, input_shape)
		"""
		if init:
			self.centers = self._select_centers(X)
			self.ampls = self._select_ampl(X)
			self.centers_0, self.ampls_0 = self.centers, self.ampls
		else:
			self.centers = np.mean(np.array([self._select_centers(X), self.centers_0]), axis=0)
			self.ampls = np.mean(np.array([self._select_ampl(X), self.ampls_0]), axis=0)
		G = self._calculate_interpolation_matrix(X)
		self.weights = np.dot(np.linalg.pinv(G), Y)

	def predict(self, X):
		"""
		# Arguments
			X: test data
		# Input shape
			(num_test_samples, input_shape)
		"""
		G = self._calculate_interpolation_matrix(X)
		predictions = np.dot(G, self.weights)
		return predictions

	def scf(self, X, Y, conv_min=10e-6, max_iter=400):

		for i in range(max_iter):

			if not i:
				self.fit(X=X, Y=Y)
			else:
				self.fit(X=X, Y=Y, init=False)


			y_pred = self.predict(X=X)
			mse = (np.square(Y - y_pred)).mean(axis=None)
			print("{} {:2.4}".format(i + 1,mse))
			if mse <= conv_min:
				print("Converged!")
				return y_pred
				break
		print("Not Converged!")
		return y_pred


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	# example IV
	if 1:
		# generating data
		x = np.linspace(0, 10, 100)
		y = np.sin(x) + np.cos(x) ** 2

		# fitting RBF-Network with data
		model = RBFN(hidden_shape=10, sigma=1.)

		# model.fit(x, y)

		y_pred = model.scf(x, y)

		# plotting 1D interpolation
		plt.plot(x, y, 'b-', label='real')
		plt.plot(x, y_pred, 'r-', label='fit')
		plt.legend(loc='upper right')
		plt.title('Interpolation using a RBFN')

		plt.show()

	# example I
	if 0:
		# generating data
		x = np.linspace(0, 10, 100)
		y = np.sin(x) + np.cos(x) ** 2

		# fitting RBF-Network with data
		model = RBFN(hidden_shape=10, sigma=1.)

		model.fit(x, y)

		y_pred = model.predict(x)

		# plotting 1D interpolation
		plt.plot(x, y, 'b-', label='real')
		plt.plot(x, y_pred, 'r-', label='fit')
		plt.legend(loc='upper right')
		plt.title('Interpolation using a RBFN')

		plt.show()

	# example II

	if 0:
		# generating data
		x = np.linspace(0, 10, 100)
		y = np.sin(x) + np.cos(x) ** 2

		# fitting RBF-Network with data
		model = RBFN(hidden_shape=10, sigma=1.)
		for i in range(5):
			model.fit(x, y)

			y_pred = model.predict(x)

			# plotting 1D interpolation
			plt.plot(x, y, 'b-', label='real')
			plt.plot(x, y_pred, 'r-', label='fit')
			plt.legend(loc='upper right')
			plt.title('Interpolation using a RBFN')

		plt.show()

	# generating dummy data for interpolation

	# example III
	if 0:
		x, y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
		z = (np.sin(np.sqrt((x - 2.) ** 2 + (y - 1) ** 2)) -
		     np.sin(np.sqrt((x + 2.) ** 2 + (y + 4) ** 2))) / 2.

		# fitting RBF-Network with data
		features = np.asarray(list(zip(x.flatten(), y.flatten())))
		model = RBFN(hidden_shape=70, sigma=1.)
		model.fit(features, z.flatten())
		predictions = model.predict(features)

		# plotting 2D interpolation
		figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
		figure.suptitle('RBF-Network 2D interpolation', fontsize=20)
		axis_right.set_title('fit', fontsize=20)
		axis_left.set_title('real', fontsize=20)
		axis_left.contourf(x, y, z)
		right_image = axis_right.contourf(x, y, predictions.reshape(20, 20))
		plt.show()

