import unittest

import numpy as np

from RBFN import *


def generate_genetic(grid):
    # generating data
    x = np.linspace(-5, 5, grid)
    y = np.sin(x) + np.cos(x) ** 2 + np.random.uniform(-0.1, 0.1, grid)
    # Just check if everything is running and connected, but no extended test
    # for travis-CI
    gens = GeneticFitter.GeneticOpt(x, y, items=1, epochs=1, max_iter=1,
                                    conv_min=10e-9, hidden_ranges=[5, 45],
                                    method=['gaus'])
    y_pred = gens.evolver()


def generate_genetic_extended(grid):
    # generating data
    x = np.linspace(-1, 1, grid)
    y = np.sin(x) + np.cos(x) ** 2 + np.random.uniform(-0.1, 0.1, grid)
    # Just check if everything is running and connected, but no extended test
    # for travis-CI
    gens = GeneticFitter.GeneticOpt(x, y, items=1, epochs=1, max_iter=1,
                                    conv_min=10e-9, hidden_ranges=[5, 45],
                                    method=['gaus', 'lorz', 'psed'])
    y_pred = gens.evolver()


def generate_rbfn(grid, iter_):
    # generate data
    x = np.linspace(-10, 10, grid)
    y = np.sin(x) + np.cos(x) ** 2 + np.random.uniform(-0.1, 0.1, grid)

    # fitting RBF-Network with data
    model = RBFNetwork.RBFN(hidden_shape=15)
    y_pred, mse, params, conv = model.scf(x, y, max_iter=iter_)


class GeneticTests(unittest.TestCase):
    def test_size_1(self):
        generate_genetic(grid=50)

    def test_size_2(self):
        generate_genetic_extended(grid=25)

    def test_size_3(self):
        generate_rbfn(grid=50, iter_=15)

    def test_size_4(self):
        generate_rbfn(grid=100, iter_=20)


if __name__ == "__main__":
    unittest.main()
