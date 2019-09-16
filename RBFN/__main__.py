import argparse
import numpy as np
import matplotlib.pyplot as plt
from RBFN import *


def cmd():
	parser = argparse.ArgumentParser(description='Process Command Line Interface for the genetic optimized fitting of'
	                                             'Radial Basis Function Networks')

	parser.add_argument("infile", help="Input data file")
	parser.add_argument("-p", "--para", nargs=3, default=[10,10,50], type=int,
	                    help="Number of items, evolutions, iterations (default: 10, 10, 50)")
	parser.add_argument("-hl", "--hidden", nargs=2, default=[10, 50], type=int,
	                    help="Range of hidden-layers (default: 10 to 50)")
	parser.add_argument("-mt", "--mutation", nargs=2, default=[10, 50], type=int,
	                    help="Mutation-level (default: 0.1 0.1)")
	parser.add_argument("-i", "--incest",  action="store_true", default=False,
	                    help="Activating the incest optiont")
	parser.add_argument("-m", "--method", nargs='?', default=['gaus', 'lorz'], type=str,
	                    help="Range of hidden-layers (default: 10 to 50)")
	parser.add_argument("--plot", action="store_true", default=False,
	                    help="Create plots during the optimization cylce")

	args = parser.parse_args()

	fh = open(args.infile, "r")
	x, y = np.genfromtxt(args.infile)
	gens = GeneticFitter(x, y, items=args.para[0], epochs=args.para[1], max_iter=args.para[2],hidden_ranges=args.hidden,
	              method=args.method, plot=args.plot)
	ypred = gens.envolver()

if __name__ == '__main__':

	cmd()

