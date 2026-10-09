from adjustText import adjust_text
import seaborn as sns
import pandas as pd
import numpy as np


def plot_volcano(
    data: pd.DataFrame,
    logfc: str,
    pvals: str,
    ax,
    logfc_min: float = 1.00,
    pvals_min: float = 1e-2,
    text_size: int = 5,
    show_top: int = None,
    stat: str = 'stat',
    hue: str = None,
):
    sig = ~data[logfc].between(-logfc_min, logfc_min) & data[pvals].lt(pvals_min)

    named_genes = data.index[sig]
    if show_top is not None:
        sorted_data = data[stat].sort_values()
        neg = sorted_data[sorted_data.lt(0)].head(show_top).index
        pos = sorted_data[sorted_data.gt(0)].tail(show_top).index
        named_genes = named_genes.intersection(pos.union(neg))

    scatter_kw = dict(x=logfc, y=pvals, s=5, lw=0, legend=False)
    sns.scatterplot(data[~sig], color='lightgray', ax=ax, **scatter_kw)
    sns.scatterplot(data[sig], ax=ax, **scatter_kw)

    line_kw = dict(lw=0.5, linestyle='--', color='k')
    objects = [
        ax.axhline(pvals_min, **line_kw),
        ax.axvline(-logfc_min, **line_kw),
        ax.axvline(logfc_min, **line_kw),
    ]

    texts = [
        ax.text(data.loc[gene, logfc], data.loc[gene, pvals], gene,
                ha='center', va='center', size=text_size)
        for gene in named_genes
    ]

    ax.set_yscale('log')

    pv = data[pvals].clip(lower=1e-300)
    yvals = np.log10(pv)
    h = (yvals.max() - yvals.min()) * 0.05
    ax.set_ylim(np.power(10, yvals.max() + h), np.power(10, yvals.min() - h))

    w = (data[logfc].max() - data[logfc].min()) * 0.05
    ax.set_xlim(data[logfc].min() - w, data[logfc].max() + w)

    adjust_text(
        texts,
        objects=objects,
        ax=ax,
        force_text=(0.2, 0.2),
        force_explode=(0.125, 0.125),
        min_arrow_len=0,
        expand=(1.1, 1.1),
        linewidth=0.,
        arrowprops={'color': 'tab:green', 'lw': 0.125, 'shrinkB': 1, 'shrinkA': 1},
    )
