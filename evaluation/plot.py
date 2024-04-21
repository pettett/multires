import matplotlib.pyplot as plt


PLOT_SIZE = (7, 3)


def config(ax: plt.Axes, fig: plt.Figure):

    if fig:
        fig.set_size_inches(PLOT_SIZE[0], PLOT_SIZE[1])
    if ax:
        ax.legend()
