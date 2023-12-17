import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

from plot.colors import Neon


def plot_with_intervals(min_data, max_data, mean_data, inds=None, label=None, c=Neon.RED.norm, lw=3, linestyle='-', dashes=None, ax=None):
    ax = sns.lineplot(x=inds, y=mean_data, color=c, linewidth=lw, label=label, linestyle=linestyle, dashes=dashes, ci=95, ax=ax)
    ax.fill_between(inds, y1=min_data, y2=max_data, alpha=0.1, color=c)
    ax.lines[-1].set_linestyle(linestyle)
    ax.legend(handlelength=3)
    if dashes is not None:
        ax.lines[-1].set_dashes(dashes)
    return ax


def plot_with_error_bars(dev_data, mean_data, inds=None, label=None, c=Neon.RED.norm, lw=3, linestyle='-', dashes=None, ax=None):
    ax = sns.lineplot(x=inds, y=mean_data, color=c, linewidth=lw, label=label, linestyle=linestyle, dashes=dashes, ax=ax)
    ax.errorbar(inds, mean_data, dev_data, color=c, linewidth=lw, label=label, dashes=dashes)
    ax.lines[-1].set_linestyle(linestyle)
    ax.legend(handlelength=3)
    if dashes is not None:
        ax.lines[-1].set_dashes(dashes)
    return ax


def style_ax(ax, ac, tc, move_right=False, dashed=False):
    ax.title.set_color(ac)
    ax.xaxis.label.set_color(ac)
    ax.yaxis.label.set_color(ac)
    if move_right:
        ax.spines['left'].set_position(('axes', 1.015))
    if dashed:
        ax.spines['left'].set_linestyle((1, (1, 2)))
    for spine in ax.spines.values():
        spine.set_color(tc)
        spine.set_linewidth(2)


def fig():
    fig = plt.gcf()
    fig.set_size_inches(7, 5, forward=True)


def save(fPath):
    fig = plt.gcf()
    fig.savefig(fPath, dpi=250, bbox_inches='tight', pad_inches=0)


def process_axes(axes, xticks=None, log=False, ylim=None, xlabel=None):
    font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)
    ts = 20
    tc = Neon.BLACK.norm
    pad = 5

    for ax in axes:
        ax.yaxis.reset_ticks()
        if xticks is not None:
            ax.xaxis.set_ticks(xticks)
        if log:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xlabel is not None:
            ax.set_xlabel(xlabel, color=tc, fontdict=font)

        ax.tick_params(axis='y', labelsize=ts, labelcolor=tc, direction='out', pad=pad)
        ax.tick_params(axis='x', labelsize=ts, labelcolor=tc)
        labels = ax.xaxis.get_majorticklabels()#  + ax.yaxis.get_majorticklabels()
        for label in labels:
            label.set_fontweight('bold')

        # ----- style ------
        style_ax(ax, Neon.BLACK.norm, Neon.BLACK.norm, False)
    fig()
