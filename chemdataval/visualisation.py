import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .utils import data_stats
import matplotlib


def fivefigs(figsize=(24, 16), yaxis_label=None, shareaxes=True, ylim=None):
    """
    Helper function to generate five figures in my preferred layout.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=2, ncols=6)

    ax11 = fig.add_subplot(gs[0, :2])
    ax12 = fig.add_subplot(gs[1, 1:3])
    if shareaxes:
        ax21 = fig.add_subplot(gs[0, 2:4], sharey=ax11)
        ax31 = fig.add_subplot(gs[0, 4:6], sharey=ax11)
        ax22 = fig.add_subplot(gs[1, 3:5], sharey=ax12)
    else:
        ax21 = fig.add_subplot(gs[0, 2:4])
        ax31 = fig.add_subplot(gs[0, 4:6])
        ax22 = fig.add_subplot(gs[1, 3:5])

    axs = [ax11, ax21, ax31, ax12, ax22]

    if yaxis_label is not None:
        ax11.set_ylabel(yaxis_label, fontsize=32)
        ax12.set_ylabel(yaxis_label, fontsize=32)

    if ylim is not None:
        [ax.set_ylim(*ylim) for ax in axs]

    return fig, axs


def visualise_learning(results, ax=None, fold=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if fold is not None:
        ax.set_title(f"Fold {fold}", fontsize=36)

    stats = {}
    for name, result in results.items():
        if name == "Test indexes":
            continue
        stats[name] = data_stats(result["scores"])

    for name in stats.keys():
        if name == "Test indexes":
            continue
        mean, var = stats[name]
        ax.plot(mean, label=name.capitalize())
        ax.fill_between(range(len(mean)), mean - var, mean + var, alpha=0.2)

    ax.legend(frameon=False, fontsize=28, loc="lower right")
    ax.set_ylabel("$R^2$", fontsize=32)
    ax.set_xlabel("Queries", fontsize=32)

    return ax


def visualise_kf(results, ylim=(0, 1)):
    """
    Simple wrapper function around visualise learning.
    """
    fig, axs = fivefigs()

    for idx, result in enumerate(results):
        ax = axs[idx]
        visualise_learning(results[f"Fold {idx+1}"], ax=ax, fold=idx + 1)
        if ylim is not None:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    return fig


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)


def plot_convergence(inform, train_idxs=None, ax=None):
    """
    Plots the convergence of the data Shapley values over time.

    Parameters
    ----------
    inform : np.array(N x M)
        N x M where N is the number of runs, and M the number of samples.

    train_idxs : np.ndarray, default=None
        The indices of the sample to plot, typically just the training indices. If None, then the entire dataset is plotted

    ax : matplotlib.axes._subplots.AxesSubplot, default=None
        The matplotlib axis to plot on. If None, a new plot is generated.
    """
    inform = np.ma.masked_equal(inform, 0)

    assert (
        len(inform.shape) == 2
    ), "Inform must be two dimensional, N x M where N is the number of runs, and M the number of samples."

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    if train_idxs is None:
        train_idxs = np.arange(inform.shape[1])

    ax.plot(
        range(len(inform[:, train_idxs])),
        [inform[:x, train_idxs].mean(axis=0) for x in range(len(inform))],
        linewidth=0.5,
        color="k",
        alpha=0.25,
    )

    return ax


def plot_multiple_runs(
    sizes, data, color="green", label=None, ax=None, alpha=1.0, linestyle=None
):
    """
    Function to plot individual sampling runs.

    Parameters
    ----------
    sizes : iter
        An iterable containing each of the sample sizes tested

    data : [[float]]
        A 2D array in which the first dimension is the sample size, and the second is the scores across each run

    color : str, default="green"
        The color to make the line and the fill region around it

    label : str, default=None
        The label to provide to the plotted line.

    ax : matplotlib.Axes, default=None
        The matplotlib axis to plot on

    alpha : 0 <= float <= 1, default=1.0
        The alpha value. Note that the fill region will be given an alpha value 1/5 of the value provided in this argument.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    data = ensure_array(data)

    ax.plot(
        sizes,
        np.mean(data, axis=1),
        linestyle=linestyle,
        color=color,
        label=label,
        alpha=alpha,
    )
    ax.fill_between(
        sizes,
        np.mean(data, axis=1) - np.sqrt(np.var(data, axis=1)),
        np.mean(data, axis=1) + np.sqrt(np.var(data, axis=1)),
        color=color,
        alpha=alpha * 0.2,
    )

    return ax


def plot_comparison(sizes, informativeness_data, random_data, bad_data=None, ax=None):
    informativeness_data, random_data = (
        ensure_array(informativeness_data),
        ensure_array(random_data),
    )

    ax.plot(
        sizes, np.mean(informativeness_data, axis=1), color="green", label="Informative"
    )
    ax.fill_between(
        sizes,
        np.mean(informativeness_data, axis=1)
        - np.sqrt(np.var(informativeness_data, axis=1)),
        np.mean(informativeness_data, axis=1)
        + np.sqrt(np.var(informativeness_data, axis=1)),
        color="green",
        alpha=0.2,
    )
    ax.plot(sizes, np.mean(random_data, axis=1), "--", color="k", label="Random")
    ax.fill_between(
        sizes,
        np.mean(random_data, axis=1) - np.sqrt(np.var(random_data, axis=1)),
        np.mean(random_data, axis=1) + np.sqrt(np.var(random_data, axis=1)),
        color="k",
        alpha=0.2,
    )
    if bad_data is not None:
        ax.plot(sizes, bad_data, color="red", label="Uninformative")

    return ax
