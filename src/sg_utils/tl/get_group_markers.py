from sklearn.linear_model import RANSACRegressor as regressor
from matplotlib import pyplot as plt, ticker
from adjustText import adjust_text
import scanpy as sc
import pandas as pd
import numpy as np


def _get_markers(
    ad: sc.AnnData,
    groupby: str,
    name: str,
    max_residual: float = 0.5,
    n_pts: int = 20,
    n_bins: int = None,
    min_difference: float = 1.0,
    plot: bool = False,
    highlight: dict = None,
    ax = None,
):
    # Get group
    group = ad.obs.groupby(groupby).get_group(name)
    
    # Get means
    diff = ad.obs.index.difference(group.index)
    s1 = len(group.index)
    u1_vals = ad[group.index, :].X.sum(0).A.flatten() / s1 # group mean
    u1 = pd.Series(u1_vals, ad.var.index)
    s2 = len(diff)
    u2_vals = ad[diff, :].X.sum(0).A.flatten() / s2 # other mean
    u2 = pd.Series(u2_vals, ad.var.index)

    # Log-transform
    mask = u1.eq(0) & u2.eq(0)
    u1 = u1[~mask]
    u2 = u2[~mask]
    l1 = np.log10(u1, where=u1.values!=0)
    l2 = np.log10(u2, where=u2.values!=0)

    # Select points for regression
    yi = l1 - l2
    if n_bins is not None:
        bins = yi.groupby(pd.cut(l2, n_bins))
        max_diff = bins.nlargest(n_pts).index.get_level_values(1)
    else:
        max_diff = yi.nlargest(n_pts).index

    # Robust linear model
    X = l2.loc[max_diff].values.reshape(-1,1)
    y = l1.loc[max_diff]
    model = regressor().fit(X, y)
    yp = model.predict(l2.values.reshape(-1,1))
    residuals = l1 - yp
    markers = l2.index[
        residuals.abs().le(max_residual) & \
        yi.ge(min_difference)
    ]

    if plot:
        xr = np.logspace(l2.min(), l2.max())
        yr = np.power(10, model.predict(np.log10(xr.reshape(-1,1))))
        yr -= max_residual
        mask = (xr > 0) & (yr > 0)
        _plot_markers(
            u1, u2, xr[mask], yr[mask], markers, max_diff,
            highlight=highlight,
            ax=ax,
        )

    return markers


def _plot_markers(
    u1, u2,
    xr, yr,
    markers,
    max_diff,
    s=1.5,
    highlight=None,
    ax=None,
):
    # Figure
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    # Plot all gene expression
    idx = u2.index.difference(max_diff.union(markers))
    ax.scatter(u2[idx], u1[idx], s=s, c=[0.75]*3, lw=0, zorder=1)

    # Plot potential contaminants
    idx = markers.difference(max_diff)
    ax.scatter(u2[idx], u1[idx], s=s, c='darkred', lw=0, zorder=2)

    # Plot genes used for regression
    ax.scatter(u2[max_diff], u1[max_diff], s=s, c='k', lw=0, zorder=3)

    # Plot robust linear regression 
    ax.plot(xr, yr, lw=0.5, linestyle='--', c='k')

    # Plot highlighted genes
    texts = []
    scatters = []
    if highlight is not None:
        for name, color in highlight.items():
            c = ax.scatter(u2[name], u1[name], s=s*2, lw=0, c=color, zorder=4)
            styles = dict(ha='center', va='center', size=4)
            t = ax.text(u2[name], u1[name], s=name, **styles)
            texts.append(t)

    # Formatting
    vmax = np.log10(max(u1.max(), u2.max()))
    vmin = np.log10(min(u1.min(), u2.min()))
    w = vmax - vmin
    vmin = 10**(vmin-w*0.05)
    vmax = 10**(vmax+w*0.05)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.tick_params(which='major', width=0.75, length=4, labelsize=6)
    ax.tick_params(which='minor', width=0.25, length=2, labelsize=6)

    # Adjust labels to not overlap
    adjust_text(
        texts,
        force_text=(0.25, 0.5),
        force_explode=(0.125, 0.25),
        min_arrow_len=0,
        expand=(1.1,1.1),
        arrowprops={'color':'k', 'lw':0.05, 'shrinkB':1, 'shrinkA':1}
    )


def get_group_markers(
    ad: sc.AnnData,
    groupby: str,
    max_residual: float = 0.5,
    n_pts: int = 20,
    n_bins: int = None,
):
    for name in ad.obs[groupby].unique():

        # Get representative markers
        markers = _get_markers(ad, groupby, name, max_residual, n_pts, n_bins)
        column = name + ' Marker'

        # Update AnnData
        ad.var[column] = ad.var.index.isin(markers)
        ad.var[column] = ad.var[column].fillna(False).astype(bool)