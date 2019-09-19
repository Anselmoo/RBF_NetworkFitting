__all__ = ['plot_final', 'plot_mutate', 'plot_selection']
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def scf_run(frames):
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')

    def init():
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 0.3)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(frame)
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    plt.show()


def plot_selection(X, y, y_pred, plot=False):
    if plot:
        fig = plt.gcf()
        fig.canvas.set_window_title('Selection')
        plt.plot(X, y, 'b-', label='real')
        plt.plot(X, y_pred, 'r--', label='fit')
        plt.legend(loc='best')
        plt.title('Interpolation using a genetic-optimized RBFN')
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()


def plot_mutate(X, y, y_pred, plot=False):
    if plot:
        fig = plt.gcf()
        fig.canvas.set_window_title('Mutation')
        plt.plot(X, y, 'b-', label='real')
        plt.plot(X, y_pred, 'm--', label='fit')
        plt.legend(loc='best')
        plt.title('Interpolation using a genetic-optimized RBFN')
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()


def plot_final(X, y, y_pred, plot=False):
    if plot:
        fig = plt.gcf()
        fig.canvas.set_window_title('Final-Result')
        plt.plot(X, y, 'b-', label='real')
        plt.plot(X, y_pred, 'm--', label='fit')
        plt.legend(loc='best')
        plt.title('Interpolation using a genetic-optimized RBFN')
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()
