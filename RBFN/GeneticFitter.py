import numpy as np

from RBFN import *


class GeneticOpt(object):
    """
    Core-design of the class-layout was adapted from Oarriaga:
    https://github.com/oarriaga/RBF-Network according to his licencse
    and further modified
    """

    def __init__(self, X, y, hidden_ranges=[1, 25],
                 method=['gaus', 'lorz', 'psed'], items=10, epochs=10,
                 max_iter=50,
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
        """
        Generating a random-guess for the given broaderings and models

        Returns
        -------
        object-array:
                With floats for sigmas and strings for models
        """
        # Random Guess for the Hidden-Layers
        chrom_1 = np.random.randint(low=self.hidden_ranges[0],
                                    high=self.hidden_ranges[1],
                                    size=self.items,
                                    dtype=np.int)
        # Random Guess for the kind of method
        chrom_2 = np.random.choice(self.method, size=self.items)
        return np.array([chrom_1, chrom_2], dtype=np.object)

    @staticmethod
    def choose(choices):
        p = np.random.uniform(0, choices[-1])
        return np.abs(choices - p).argmin()

    def _selection(self, generation):
        """
        Based on the initial generation (guess) new mommy and daddy pairs
        will be generated

        Parameters
        ----------
        generation: array-object
                With floats for sigmas and strings for models

        Returns
        -------
        next-generation: array-object
                With floats for sigmas and strings for models
        """
        length_generations = generation.shape[1]
        range_generations = range(length_generations)

        mse_init = np.zeros(length_generations)
        for i in range_generations:
            print('\n-------------------------------------------------')
            print("Init-Items #{} of {}".format(i + 1,
                                                length_generations))
            if not self.conv_status:
                self.hidden_shape, self.mode = generation[0, i], generation[
                    1, i]
                print('-------------------------------------------------')
                print("Number of Functions: {}, Kind of Functions: {}".format(
                    self.hidden_shape, self.mode))
                print('-------------------------------------------------')
                netw = RBFNetwork.RBFN(hidden_shape=self.hidden_shape,
                                       mode=self.mode)
                self.y_pred, self.mse, self.params, self.conv_status = \
                    netw.scf(
                        X=self.X, y=self.y,
                        max_iter=self.max_iter,
                        conv_min=self.conv_min)
                PlotResults.plot_selection(X=self.X, y=self.y,
                                           y_pred=self.y_pred, plot=self.plot)
                mse_init[i] = self.mse
            else:
                break

        next_generation = np.zeros_like(generation, dtype=np.object)
        for i in range_generations:
            if not self.conv_status:
                print("\t\nMixing-Items #{} of {}".format(i + 1,
                                                          length_generations))
                next_generation[0, i] = generation[
                    0, self.choose(mse_init)]  # Mummy
                next_generation[1, i] = generation[
                    1, self.choose(mse_init)]  # Daddy
            else:
                break
        return next_generation

    def _mutate(self, generation, mutate=[.1, .1]):
        """
        Mutation of the genotype based on the mutation level for both
        chromosomes

        Parameters
        ----------
        generation: array-object
                With floats for sigmas and strings for models
        mutate: float-list
                Mutation probability
        """
        length_generations = generation.shape[1]
        range_generations = range(length_generations)
        for i in range_generations:
            print("\t\nMutation-Items #{} of {}".format(i + 1,
                                                        length_generations))
            # Evolution Asking - I for number of hidden-layers
            if np.random.rand() < mutate[0]:
                generation[0, i] = \
                    np.random.randint(low=self.hidden_ranges[0],
                                      high=self.hidden_ranges[1], size=1,
                                      dtype=np.int)[0]
            # Evolution Asking - II for kind of model
            if np.random.rand() < mutate[1]:
                generation[1, i] = np.random.choice(self.method, size=1)[0]

            # Fit- Test
            if not self.conv_status:
                self.hidden_shape, self.mode = generation[0, i], generation[
                    1, i]
                print('-------------------------------------------------')
                print("Number of Functions: {}, Kind of Function: {}".format(
                    self.hidden_shape, self.mode))
                print('-------------------------------------------------')
                netw = RBFNetwork.RBFN(hidden_shape=self.hidden_shape,
                                       mode=self.mode)
                self.y_pred, self.mse, self.params, self.conv_status = \
                    netw.scf(
                        X=self.X, y=self.y,
                        max_iter=self.max_iter,
                        conv_min=self.conv_min)

                PlotResults.plot_mutate(X=self.X, y=self.y,
                                        y_pred=self.y_pred,
                                        plot=self.plot)
            else:
                break

    def evolver(self, mutate=[0.1, 0.1], incest=False):
        """
        Evolution of the genotype (hyper-parameter for the radial basis
        function network)

        Parameters
        ----------
        mutate: float-list
                Mutation probability
        incest: bool
                Using old genotype
        """
        generation = self._random_guess()
        next_generation = np.zeros_like(generation, dtype=np.object)

        for i in range(self.epochs):
            print('---------------------------------------------------')
            print("\t\t\t|Evolution #{} of {}|".format(i + 1, self.epochs))
            print('---------------------------------------------------\n\n')

            if incest:  # Generation new genotype based on old genotype
                if not i:
                    next_generation = self._selection(generation)
                else:
                    next_generation = self._selection(next_generation)
            else:
                next_generation = self._selection(generation)

            if not self.conv_status:
                self._mutate(next_generation, mutate=mutate)
                if self.conv_status:
                    print("Converged!")
                    break
            else:
                print("Converged!")
                break
        # print(self.mse)
        return self.y_pred

    def __call__(self, *args, **kwargs):
        print("__call__")


# RBFNetwork.RBFN().__init__()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 250)
    y = np.sin(x) + np.cos(x) ** 2 + np.random.uniform(-0.1, 0.1, 250)
    gens = GeneticOpt(x, y, items=10, epochs=4, max_iter=10, conv_min=10e-4,
                      hidden_ranges=[5, 145], method=['gaus'],
                      plot=True)
    y_pred = gens.evolver()
    print(gens.__dict__)
    # plotting 1D interpolation
    plt.plot(x, y, 'b-', label='real')
    plt.plot(x, y_pred, 'r-', label='fit')
    plt.legend(loc='best')
    plt.title('Interpolation using a genetic-optimized RBFN')

    plt.show()
