import argparse

import numpy as np

from RBFN import *


def report(dstream):
	"""
	Plotting the final-reasults
	"""
	val = dstream.__dict__
	print('\n---------------------------------------------------')
	print("Final Results")
	print('---------------------------------------------------')
	print("Number of Shapes -> ", np.unique(val['hidden_shape'])[0])
	print("Kind of Shape -> ", np.unique(val['mode'])[0])
	print("Sigma of Gaussian -> {:2.5}".format(np.unique(val['params'][0])[0]))
	print("Sigma of Lorentzian -> {:2.5}".format(np.unique(val['params'][1])[0]))
	print("Alpaha -> {:2.5}".format(np.unique(val['params'][2])[0]))
	print("Mean Square Error -> {:2.5}".format(np.unique(val['mse'])[0]))


def cmd():
	"""
	Command Line Interface for reading data from the cmd
	"""
	parser = argparse.ArgumentParser(description='Process Command Line Interface for the genetic optimized fitting of'
	                                             'Radial Basis Function Networks')

	parser.add_argument("infile", help="Input data file")
	parser.add_argument("-p", "--para", nargs=3, default=[10, 10, 50], type=int,
	                    help="Number of items, evolutions, iterations (default: 10, 10, 50)")
	parser.add_argument("-hl", "--hidden", nargs=2, default=[10, 50], type=int,
	                    help="Range of hidden shapes (default: 10 to 50)")
	parser.add_argument("-mt", "--mutate", nargs=2, default=[0.1, 0.1], type=float,
	                    help="Mutation-level (default: 0.1 0.1)")
	parser.add_argument("-i", "--incest", action="store_true", default=False,
	                    help="Activating the incest optiont")
	parser.add_argument("-m", "--method", nargs='+', default=['gaus', 'lorz', 'psed'], type=str,
	                    help="Range of methods (default: all three ['gaus', 'lorz', 'psed'] ")
	parser.add_argument("-pl", "--plot", action="store_true", default=False,
	                    help="Create plots during the optimization cylce")

	args = parser.parse_args()

	data = np.genfromtxt(args.infile)
	x, y = data[:, 0], data[:, 1]

	# Loading the genetic optimizer
	gens = GeneticFitter.GeneticOpt(x, y, items=args.para[0], epochs=args.para[1], max_iter=args.para[2],
	                                hidden_ranges=args.hidden,
	                                method=args.method, plot=args.plot)
	ypred = gens.evolver(mutate=args.mutate, incest=args.incest)

	np.savetxt(args.infile.rsplit('.', 1)[0] + '.export', np.array([x, y, ypred]).T, delimiter='\t',
	           header='X\ty\typred')
	PlotResults.plot_final(x, y, ypred, plot=args.plot)
	report(dstream=gens)


if __name__ == '__main__':
	cmd()
