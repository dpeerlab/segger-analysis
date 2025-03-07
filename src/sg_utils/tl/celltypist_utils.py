import scipy as sp
import scanpy as sc
import celltypist as ct
from sg_utils.pp.preprocess_rapids import HiddenPrints


def build_celltypist_model(
    ad_atlas: sc.AnnData,
    celltype_col: str, 
    raw_layer: str,
    target_sum: int = 1000,
    sample_size: int = 5000,
):
    # Re-normalize counts to 10K total
    t = target_sum
    ad_atlas.X = ad_atlas.layers[raw_layer].copy()
    sc.pp.downsample_counts(ad_atlas, counts_per_cell=t)
    ad_atlas.layers[f'norm_{t}'] = ad_atlas.X.copy()
    sc.pp.normalize_total(ad_atlas, layer=f'norm_{t}', target_sum=t)
    
    # Logarthmize
    ad_atlas.layers[f'lognorm_{t}'] = ad_atlas.layers[f'norm_{t}'].copy()
    if 'log1p' in ad_atlas.uns:
        del ad_atlas.uns['log1p']
    sc.pp.log1p(ad_atlas, layer=f'lognorm_{t}')
    
    # Subsample using more granular cell types (to not lose any one cell type)
    # But transfer labels using the compartment labels
    gb = ad_atlas.obs.groupby(celltype_col, observed=True)
    sample = gb.sample(sample_size, replace=True).index.drop_duplicates()
    
    # Predict on log counts
    ad_atlas.X = ad_atlas.layers[f'lognorm_{t}']
    with HiddenPrints():
        ct_model = ct.train(
            ad_atlas[sample],
            labels=celltype_col,
            use_GPU=True,
            check_expression=False,
    )
    return ct_model


def annotate_cell_types(
    ad: sc.AnnData,
    ct_model: ct.Model,
    cluster_col: str = 'phenograph_cluster',
    target_sum: int = 1000,
    suffix: str = None,
):
    # Re-normalize consistent with CellTypist model
    t = target_sum
    ad.layers[f'norm_{t}'] = ad.raw.X.copy()
    sc.pp.normalize_total(ad, layer=f'norm_{t}', target_sum=t)
    ad.layers[f'lognorm_{t}'] = ad.layers[f'norm_{t}'].copy()
    if 'log1p' in ad.uns: del ad.uns['log1p']
    sc.pp.log1p(ad, layer=f'lognorm_{t}')
    
    # Cell type
    with HiddenPrints():
        ad.X = ad.layers[f'lognorm_{t}']
        preds = ct.annotate(
            ad, model=ct_model, majority_voting=True,
            over_clustering='phenograph_cluster',
            min_prop=0.2,
        )
    
    # Label AnnData
    suffix = '' if suffix is None else f'_{suffix}'
    ad.obs[f'celltypist_label{suffix}'] = preds.predicted_labels['predicted_labels']
    ad.obs[f'celltypist_label_cluster{suffix}'] = preds.predicted_labels['majority_voting']
    ad.obs[f'celltypist_probability{suffix}'] = preds.probability_matrix.max(1)
    for col in preds.probability_matrix.columns:
        ad.obs[f'{col}_probability'] = preds.probability_matrix[col]
    entropy = sp.stats.entropy(preds.probability_matrix, axis=1)
    ad.obs[f'celltypist_entropy{suffix}'] = entropy
    
    # Cleanup
    del ad.layers[f'lognorm_{t}'], ad.layers[f'norm_{t}']
