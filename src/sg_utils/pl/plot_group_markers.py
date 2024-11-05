from sg_utils.tl.get_group_markers import _get_markers
import scanpy as sc

def plot_group_markers(
    ad: sc.AnnData,
    groupby: str,
    name: str,
    max_residual: float = 0.5,
    n_pts: int = 20,
    n_bins: int = None,
    highlight: dict = None,
    ax = None,
):

    # Plot representative markers
    _get_markers(
        ad,
        groupby=groupby,
        name=name,
        max_residual=max_residual,
        n_pts=n_pts,
        n_bins=n_bins,
        highlight=highlight,
        ax=ax,
        plot=True,
    )