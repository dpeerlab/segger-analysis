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
    hue: str = None,
):
    # Whether to bolden and show name
    sig = ~data[logfc].between(-logfc_min, logfc_min) 
    sig &= data[pvals].lt(pvals_min)
    named_genes = data.index[sig]

    # Scatters
    styles = dict(x=logfc, y=pvals, s=8, lw=0, legend=False)
    sns.scatterplot(data[~sig], color='lightgray', **styles)
    sns.scatterplot(data[sig], **styles)
    
    # Vlines
    styles = dict(lw=0.5, linestyle='--', color='k')
    h1 = ax.axhline(pvals_min, **styles)
    v1 = ax.axvline(-logfc_min, **styles)
    v2 = ax.axvline(logfc_min, **styles)
    objects = [h1, v1, v2]
    
    # Named genes
    texts = []
    for gene in named_genes:
        if gene in data[sig].index:
            row = data.loc[gene]
            styles = dict(ha='center', va='center', size=6)
            t = ax.text(row[logfc], row[pvals], gene, **styles)
            texts.append(t)
    
    # Formatting
    yvals = np.log10(data[pvals])
    h = (yvals.max() - yvals.min()) * 0.05
    ax.set_ylim(
        np.power(10, yvals.max() + h),
        np.power(10, yvals.min() - h)
    )
    ax.set_yscale('log')
    w = (data[logfc].max() - data[logfc].min()) * 0.05
    ax.set_xlim(data[logfc].min() - w, data[logfc].max() + w)
    
    adjust_text(
        texts,
        objects=objects,
        force_text=(0.25, 0.5),
        force_explode=(0.125, 0.25),
        min_arrow_len=0,
        expand=(1.1, 1.1),
        linewidth=0,
        arrowprops={'color':[0]*4, 'lw':0.0, 'shrinkB':1, 'shrinkA':1}
    )