__all__ = ['RBFN']

import numpy as np


class RBFN(object):

    def __init__(self, hidden_shape, sigma_g=1.0, sigma_l=1.0, alpha=0.5,
                 mode='gaus'):
        """
        Radial Basis function network for Gaussian, Lorentz, Pseudo-Voight
        https://en.wikipedia.org/wiki/Radial_basis_function_network

        Parameters
        ----------
        hidden_shape: int
                Number of functions in the hidden-layyer
        sigma_g: float
                Starting Sigma of the Gaussian
        sigma_l: float
                Starting Sigma of the Lorentzian
        alpha: float
                Starting mixture of Gaussian and Lorentzian Contribution
        mode: str
                Kind of Kernel-Function
        """
        self.hidden_shape = hidden_shape
        self.mode = mode
        self.sigma_g = np.full(hidden_shape, sigma_g)
        self.sigma_l = np.full(hidden_shape, sigma_l)
        self.alpha = np.full(hidden_shape, alpha)
        self.centers = None
        self.weights = None

    def _kernel_function_gaussian(self, sigma_g, center, data_point, ampl=1.):
        """
        Gaussian distribution
        https://en.wikipedia.org/wiki/Normal_distribution

        f(x\mid \mu ,\sigma ^{2})={\frac {1}{\sqrt {2\pi \sigma ^{2}}}}e^{-{
        \frac {(x-\mu )^{2}}{2\sigma ^{2}}}}
        """
        return ampl / (sigma_g * np.sqrt(2 * np.pi)) * np.exp(
            -1 / (2 * sigma_g) ** 2
            * np.linalg.norm(center - data_point) ** 2)

    def _kernel_function_lorentzian(self, sigma_l, center, data_point,
                                    ampl=1.):
        """
        Cauchy distribution
        https://en.wikipedia.org/wiki/Cauchy_distribution

        f(x;x_{0},\gamma )={1 \over \pi \gamma }\left[{\gamma ^{2} \over (
        x-x_{0})^{2}+\gamma ^{2}}\right]
        """
        return ampl / np.pi * np.divide(sigma_l, np.linalg.norm(
            center - data_point) ** 2 + sigma_l ** 2)

    def _kernel_function_pseudovoigt(self, alpha, sigma_g, sigma_l, center,
                                     data_point, ampl):
        """
        Pseudo-Voigt distribution
        https://en.wikipedia.org/wiki/Voigt_profile

        {\displaystyle V_{p}(x)=\eta \cdot L(x,f)+(1-\eta )\cdot G(x,f)} \;
        \text{with} \; 0 < \eta < 1
        """
        return (1 - alpha) * self._kernel_function_gaussian(sigma_g, center,
                                                            data_point,
                                                            ampl) \
            + alpha * self._kernel_function_lorentzian(sigma_l, center,
                                                       data_point, ampl)

    def _calculate_interpolation_matrix(self, X):
        """
        Calculates interpolation matrix using a kernel_function

        Parameters
        ---------
                X: float-array (1D)
                        Training data

        Returns
        ------
                G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                if self.mode == 'gaus':
                    G[
                        data_point_arg, center_arg] = \
                        self._kernel_function_gaussian(
                        self.sigma_g[center_arg], center,
                        data_point, self.ampls[center_arg])
                elif self.mode == 'lorz':
                    G[
                        data_point_arg, center_arg] = \
                        self._kernel_function_lorentzian(
                        self.sigma_l[center_arg], center,
                        data_point, self.ampls[center_arg])
                elif self.mode == 'psed':
                    G[
                        data_point_arg, center_arg] = \
                        self._kernel_function_pseudovoigt(
                        self.alpha[center_arg],
                        self.sigma_g[center_arg],
                        self.sigma_l[center_arg], center,
                        data_point,
                        self.ampls[center_arg])
        return G

    def _select_centers(self, X):
        """
        Initial generation of the center-guess

        Parameters
        ----------
        X: float-array
                Training data

        Returns
        -------
        centers: float-array
                Ranadom ceneters with length and shape of X
        """
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def _select_ampl(self, y):
        """
        Initial generation of the center-guess

        Parameters
        ----------
        y: float-array (1D)
                Training data

        Returns
        -------
        ampls: float-array (1D)
                Ranadom amplitudes with length and shape of y
        """
        random_args = np.random.choice(len(y), self.hidden_shape)
        ampls = y[random_args]
        return ampls

    def fit(self, X, y):
        """
        Fits weights using linear regression
        {\mathbf  {w}}={\mathbf  {G}}^{{-1}}{\mathbf  {y}}

        Parameters
        ---------
        X: float
                training samples
        y: float
                targets

        Returns
        --------
        self.weights: float
                updated weight-coefficients
        """
        self.centers = self._select_centers(X)
        self.ampls = self._select_ampl(y)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), y)

    def predict(self, X):
        """
        Prediction via Grid-Intervall

        Parameters
        ---------
        X: float
                training samples
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

    def _scf_correction(self, Y, y_pred, damp=[1., 1.]):
        """
        Self Consistent Field Convergence-Correction (SCF-CC)

        Parameters
        ----------
        Y: float
                targets
        y_pred: float
                predicted new targets
        damp: float-list
                Damping factors for gradient and difference

        Returns
        -------
        diff: float
                Numerical damped difference
        grad: float
                Numerical damped gradient
        """
        diff = damp[0] * np.subtract(Y, y_pred).mean(axis=None)
        grad = damp[1] * np.gradient(np.subtract(Y, y_pred)).mean(axis=None)
        return diff, grad

    def scf(self, X, y, conv_min=10e-3, max_iter=500, damp=[1., 1.]):
        """
        Self Consistent Field Cycle (SCF-C)

        Parameters
        ----------
        X: float
                training samples
        y: float
                targets
        conv_min: float
                convergence-limit of the SCF-Cycle
        max_iter: int
                Maximum numbers of iterations
        damp: float-list
                Damping factors for gradient and difference

        Returns:
        --------
        y_pred: float
                predicted new targets
        mse: float
                Mean squared error
        sigma_g: float as part of a list
                Sigma of the Gaussian-Distribution
        sigma_l: float as part of a list
                Sigma of the Lorentzian-Distribution
        alpha: float as part of a list
                Weighting factor between Gaussian- and Lorentzian-Distribution
        status: bool
                False for not converged, True for converged
        """
        # Init-Phase
        self.fit(X=X, y=y)
        y_pred = self.predict(X=X)

        print("#\tMSE\t\tDifference\tGradient")
        for i in range(max_iter):
            mse = (np.square(y_pred - y)).mean(axis=None)

            if mse <= conv_min:
                print("Converged!")
                return y_pred, mse, [self.sigma_g, self.sigma_l,
                                     self.alpha], True
                break
            else:
                diff, grad = self._scf_correction(y, y_pred, damp=damp)
                print("{}\t{:2.5f}\t{:2.8f}\t{:2.8f}".format(i + 1, mse, diff,
                                                             grad))
                if not np.isnan(diff) and not np.isnan(grad):
                    self.centers += -diff - grad
                    self.ampls += -diff - grad
                    if self.mode == 'gaus':
                        self.sigma_g += -diff - grad
                    elif self.mode == 'lorz':
                        self.sigma_l += -diff - grad
                    elif self.mode == 'psed':
                        self.alpha += -diff - grad
                        self.sigma_g += -diff - grad
                        self.sigma_l += -diff - grad

                    G = self._calculate_interpolation_matrix(X)
                    self.weights = np.dot(np.linalg.pinv(G), y)
                    y_pred = np.dot(G, self.weights)
        print("Not Converged!")
        return y_pred, mse, [self.sigma_g, self.sigma_l, self.alpha], False


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # example I
    if 1:
        # generating data
        x = np.linspace(-10, 10, 100)
        y = np.sin(x) + np.cos(x) ** 2 + np.random.uniform(-0.1, 0.1,
                                                           100) + x ** 3

        # fitting RBF-Network with data
        model = RBFN(hidden_shape=13, sigma_g=1., mode='psed')

        # model.fit(x, y)

        y_pred = model.scf(x, y)

        # plotting 1D interpolation
        plt.plot(x, y, 'b-', label='real')
        plt.plot(x, y_pred, 'r-', label='fit')
        plt.legend(loc='best')
        plt.title('Interpolation using a RBFN')

        plt.show()
