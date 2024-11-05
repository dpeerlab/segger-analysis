from sg_utils.pl.utils import lighten_color
from matplotlib import pyplot as plt, axes
from adjustText import adjust_text
from typing import Tuple
import pandas as pd
import numpy as np


def plot_volcano(
    results: dict,
    group: str,
    pvals_lim: Tuple[float],
    logfc_lim: Tuple[float],
    logfc_cutoff: float,
    ax: axes.Axes,
    palette,
    highlight: dict = None,
    label_cutoff: float = 1e-50,
):
    # Split limits
    pvals_min, pvals_max = pvals_lim
    logfc_min, logfc_max = logfc_lim
    
    # Get results
    groups = results['names'].dtype.names
    if group not in groups:
        raise ValueError(f"Requested group '{group}' not found in results.")
    
    # x-axis is Log-FCs
    genes = results['names'][group]
    x = results['logfoldchanges'][group]
    x = pd.Series(x, index=genes)
    
    # Transform pvals to y-axis
    ymax = -np.log10(pvals_min)
    ymin = -np.log10(pvals_max)
    pvals = results['pvals_adj'][group]
    y = -np.log10(pvals, out=np.zeros_like(pvals), where=pvals!=0)
    y = np.clip(y, 0, ymax)
    y = np.where(y==0, y.max(), y)
    y = np.power(10, y)
    y = pd.Series(y, index=genes)
    ymin = np.power(10, ymin)
    ymax = np.power(10, ymax)

    # Insignificant values
    styles = dict(s=3, lw=0)
    mask_neg = (y <= ymin) | (np.abs(x) < logfc_cutoff)
    ax.scatter(x[mask_neg], y[mask_neg], color='gray', **styles)
    
    # Significant 10x
    color = palette[list(set(groups).difference(group))[0]]
    mask_left = (y > ymin) & (x < -logfc_cutoff)
    ax.scatter(x[mask_left], y[mask_left], color=color, **styles)
    
    # Significant Segger
    color = palette[group]
    mask_right = (y > ymin) & (x > logfc_cutoff)
    ax.scatter(x[mask_right], y[mask_right], color=color, **styles)

    if highlight is not None:
        texts = []
        x_sig = x[mask_left | mask_right]
        y_sig = y[mask_left | mask_right]
        for gene, color in highlight.items():
            if gene in x_sig:
                styles = dict(s=4, lw=0.675, edgecolor=color, facecolor=[0]*4)
                ax.scatter(x_sig[gene], y_sig[gene], **styles)
                styles = dict(ha='center', va='center', size=4.5)
                if y_sig[gene] > np.power(10, -np.log10(label_cutoff)):
                    t = ax.text(x_sig[gene], y_sig[gene], s=gene, **styles)
                    texts.append(t)
    
    styles = dict(lw=0.5, linestyle='--', color='k')
    ax.axhline(ymin, **styles)
    ax.axvline(-logfc_cutoff, **styles)
    ax.axvline(logfc_cutoff, **styles)
    
    ax.set_xlim(logfc_min, logfc_max)
    ax.set_yscale('log')

    # Adjust labels to not overlap
    adjust_text(
        texts,
        force_text=(0.25, 0.5),
        force_explode=(0.125, 0.25),
        min_arrow_len=0,
        expand=(1.1,1.1),
        arrowprops={'color':'k', 'lw':0.05, 'shrinkB':1, 'shrinkA':1}
    )
