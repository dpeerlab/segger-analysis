from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
import matplotlib.colors as mc
import scanpy as sc
import pandas as pd
import numpy as np
import colorsys


def get_color_palette(
    ad: sc.AnnData,
    groupby: str,
    layer: str,
    n: float = 1,
):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cmap = mc.LinearSegmentedColormap.from_list('', colors, 255)
    
    fn = 'mean'
    coefs = sc.get.aggregate(ad, groupby, fn, layer=layer).to_df(layer=fn)
    coefs = pd.DataFrame(np.corrcoef(coefs), coefs.index, coefs.index)
    dists = squareform(pdist(coefs))
    Z = hierarchy.single(dists)
    
    order = hierarchy.dendrogram(Z, no_plot=True)['leaves']
    pdists = [dists[i, j] for i, j in zip(order[:-1], order[1:])]
    pdists = [0] + np.power(np.array(pdists), n).tolist()
    positions = np.cumsum(pdists) / np.sum(pdists)
    
    palette = pd.Series(
        dict(zip(coefs.index[order],[mc.to_hex(cmap(p)) for p in positions]))
    ).to_dict()

    return palette


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])