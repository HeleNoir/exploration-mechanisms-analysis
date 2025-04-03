from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm


# TODO see is and how figures can be set up better
def setup_figs(fig_size: (float, float) = (6.4, 4.8), font_scale: float = 1.6, fig_scale: float = 1.3):
    """
    Sets a fitting seaborn/matplotlib theme.
    """
    scaled_fig_size = (fig_size[0] * fig_scale, fig_size[1] * fig_scale)
    sns.set_theme(
        font_scale=font_scale,
        style='whitegrid',
        palette='colorblind',
        font='serif',
        context='paper',
        rc={'figure.figsize': scaled_fig_size}
    )


def setup_figs_descriptive():
    sns.set_theme(style="whitegrid",
                  font="Times New Roman",
                  font_scale=1.2,
                  context='paper',
                  palette='colorblind',
                  rc={
                      "lines.linewidth": 1,
                      "pdf.fonttype": 42,
                      "ps.fonttype": 42
                  })

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 200

    plt.tight_layout()


def setup_figs_latex():
    """
    Initializes the LaTeX/PGF backend.
    """
    # Always process .pdf files with pgf backend.
    import matplotlib.backend_bases
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

    # Set LaTeX parameters
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'text.latex.preamble': '\n'.join([r"\usepackage{libertine}", r"\usepackage{amsmath}"]),
        'pgf.preamble': '\n'.join([r"\usepackage{libertine}", r"\usepackage{amsmath}"]),
        'pgf.rcfonts': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _save_or_show(save: Optional[str | Path] = None) -> None:
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


# ---
# General descriptive plots
# ---
def plot_descriptive_lineplots(df, x_col, y_col, hue, style, save):
    setup_figs_descriptive()
    sns.lineplot(df, x=x_col, y=y_col, hue=hue, style=style)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    _save_or_show(f'{save}_{x_col}_{y_col}')
    if (y_col == 'DistanceToOptimum') | (y_col == 'DistanceToOptimum_mean'):
        ax = sns.lineplot(df, x=x_col, y=y_col, hue=hue, style=style)
        ax.set(yscale='log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        _save_or_show(f'{save}_{x_col}_{y_col}_logscale')


def plot_descriptive_summarised_lineplots(df, x_col, y_col, std, save):
    setup_figs_descriptive()
    setup_figs()
    sns.lineplot(x=df[x_col], y=df[y_col])
    if y_col in ['MinimumIndividualDistance_mean']:
        plt.ylim((-0.2, 1))
    plus_std = df[y_col].to_numpy() + df[std].to_numpy()
    minus_std = df[y_col].to_numpy() - df[std].to_numpy()
    sns.lineplot(x=df[x_col], y=plus_std, color='b')
    sns.lineplot(x=df[x_col], y=minus_std, color='b')
    plt.fill_between(df[x_col].astype(int), minus_std.astype(float), plus_std.astype(float), alpha=.3)
    _save_or_show(f'{save}_{x_col}_{y_col}')


def plot_twinaxes(df, algorithms, x_col, hue, save):
    setup_figs_descriptive()
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax2 = ax1.twinx()

    for algo in algorithms:
        subset = df[df[hue] == algo]
        ax1.plot(subset[x_col], subset['MinimumIndividualDistance_mean'], label=algo, linestyle='--', alpha=0.7)
        ax1.fill_between(subset[x_col].astype(int), subset['MinimumIndividualDistance_mean'].astype(float), alpha=0.2)

    for algo in algorithms:
        subset = df[df["Algorithm"] == algo]
        ax2.plot(subset[x_col], subset['DistanceToOptimum_mean'], linewidth=2, alpha=0.9)

    ax1.set_xlabel("Function Evaluations")
    ax1.set_ylabel("Mean Minimum Individual Distance")
    ax2.set_ylabel("Mean Distance to Optimum")

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.legend(loc="upper right", title="Algorithm")

    _save_or_show(f'{save}')

