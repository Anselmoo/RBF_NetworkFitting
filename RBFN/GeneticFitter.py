import numpy as np
from RBFN import *
from itertools import accumulate


class GeneticOpt(object):

	def __init__(self, X, y, hidden_ranges=[1, 25], method=['gaus', 'lorz', 'psed'], items=10, epochs=10, max_iter=50,
	             conv_min=10e-2, plot=False):
		# Data-Stream
		self.X, self.y = X, y
		# Initial parameters as a basis for the optimization
		# Part I - Genetic Optimizer
		self.items = items
		self.epochs = epochs
		self.hidden_ranges = hidden_ranges
		self.method = np.asarray(method, dtype=str)
		self.plot = plot
		# Part II - Radial Basis Function Network Optimizer
		self.max_iter = max_iter
		self.conv_min = conv_min
		self.conv_status = False
		# Variables to evolve
		self.hidden_shape = None
		self.mode = None
		self.mse = None
		self.y_pred = None
		self.param = None

	def _random_guess(self):
		# Random Guess for the Hidden-Layers
		chrom_1 = np.random.randint(low=self.hidden_ranges[0], high=self.hidden_ranges[1], size=self.items,
		                            dtype=np.int)
		# Random Guess for the kind of method
		chrom_2 = np.random.choice(self.method, size=self.items)
		return np.array([chrom_1, chrom_2], dtype=np.object)

	def _get_fitness(self):
		return self.mse <= self.conv_min

	@staticmethod
	def choose(choices):
		p = np.random.uniform(0, choices[-1])
		return np.abs(choices - p).argmin()

	def _selection(self, generation):

		length__of_generations = generation.shape[1]
		range_of_generations = range(length__of_generations)

		mse_init = np.zeros(length__of_generations)
		for i in range_of_generations:
			print('\n-------------------------------------------------')
			print("Init-Items #{} of {}".format(i + 1, length__of_generations))
			if not self.conv_status:
				self.hidden_shape, self.mode = generation[0, i], generation[1, i]
				print('-------------------------------------------------')
				print("Number of Functions: {}, Kind of Functions: {}".format(self.hidden_shape, self.mode))
				print('-------------------------------------------------')
				netw = RBFNetwork.RBFN(hidden_shape=self.hidden_shape, mode=self.mode)
				self.y_pred, self.mse, self.params, self.conv_status = netw.scf(X=self.X, y=self.y,
				                                                                max_iter=self.max_iter,
				                                                                conv_min=self.conv_min)
				PlotResults.plot_selection(X=self.X, y=self.y, y_pred=self.y_pred, plot=self.plot)
				mse_init[i] = self.mse
			else:
				break

		next_generation = np.zeros_like(generation, dtype=np.object)
		for i in range_of_generations:
			if not self.conv_status:
				print("\t\nMixing-Items #{} of {}".format(i + 1, length__of_generations))
				next_generation[0, i] = generation[0, self.choose(mse_init)]  # Mummy
				next_generation[1, i] = generation[1, self.choose(mse_init)]  # Daddy
			else:
				break
		return next_generation

	def _mutate(self, generation, mutate=[1, 1]):

		length__of_generations = generation.shape[1]
		range_of_generations = range(length__of_generations)
		for i in range_of_generations:
			print("\t\nMutation-Items #{} of {}".format(i + 1, length__of_generations))
			# Evolution Asking - I for number of hidden-layers
			if np.random.rand() < mutate[0]:
				generation[0, i] = \
				np.random.randint(low=self.hidden_ranges[0], high=self.hidden_ranges[1], size=1, dtype=np.int)[0]
			# Evolution Asking - II for kind of model
			if np.random.rand() < mutate[1]:
				generation[1, i] = np.random.choice(self.method, size=1)[0]

			# Fit- Test
			if not self.conv_status:
				self.hidden_shape, self.mode = generation[0, i], generation[1, i]
				print('-------------------------------------------------')
				print("Number of Functions: {}, Kind of Function: {}".format(self.hidden_shape, self.mode))
				print('-------------------------------------------------')
				netw = RBFNetwork.RBFN(hidden_shape=self.hidden_shape, mode=self.mode)
				self.y_pred, self.mse, self.params, self.conv_status = netw.scf(X=self.X, y=self.y,
				                                                                max_iter=self.max_iter,
				                                                                conv_min=self.conv_min)

				PlotResults.plot_mutate(X=self.X, y=self.y, y_pred=self.y_pred, plot=self.plot)
			else:
				break

	def evolver(self, mutate=[0.1, 0.1], incest=False):
		generation = self._random_guess()
		generation_0 = generation
		next_generation = np.zeros_like(generation, dtype=np.object)


		for i in range(self.epochs):
			print('---------------------------------------------------')
			print("\t\t\t|Evolution #{} of {}|".format(i + 1, self.epochs))
			print('---------------------------------------------------\n\n')

			if incest:
				if not i:
					next_generation = self._selection(generation)
				else:
					next_generation = self._selection(next_generation)
			else:
				next_generation = self._selection(generation)

			if not self.conv_status:
				self._mutate(next_generation,mutate=mutate)
				if self.conv_status:
					print("Converged!")
					break
			else:
				print("Converged!")
				break
			print(self.mse)
		return self.y_pred

	def __call__(self, *args, **kwargs):
		print("__call__")


# RBFNetwork.RBFN().__init__()


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	if 1:
		# generating data
		x = np.linspace(-5, 5, 250)
		y = np.sin(x) + np.cos(x) ** 2 + np.random.uniform(-0.1, 0.1, 250)
	if 0:
		data = np.genfromtxt('Iron_d4.txt')
		x, y = data[:, 0], data[:, 1]
	gens = GeneticOpt(x, y, items=10, epochs=4, max_iter=10, conv_min=10e-4, hidden_ranges=[5, 145], method=['gaus'], plot=True)
	y_pred = gens.evolver()
	print(gens.__dict__)
	# plotting 1D interpolation
	plt.plot(x, y, 'b-', label='real')
	plt.plot(x, y_pred, 'r-', label='fit')
	plt.legend(loc='best')
	plt.title('Interpolation using a genetic-optimized RBFN')

	plt.show()

# RBFNetwork.RBFN(hidden_shape=10).scf()
