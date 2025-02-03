from sg_utils.tl.phenograph_rapids import phenograph_rapids
from tqdm import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import warnings
import logging
import cupyx
import cuml
import sys
import os


# Class to suppress output from inside functions (e.g., PhenoGraph)
class HiddenPrints:
    def __init__(self, highest_level=logging.CRITICAL):
        self.highest_level=highest_level

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        self.previous_level = logging.root.manager.disable
        logging.disable(self.highest_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        logging.disable(self.previous_level)


def kneepoint(vals, xcoords=None):

    # Convert array of values to 2D vectors by position in array
    vals = np.array(vals)
    descending = vals[-1] < vals[0]
    if descending:
        vals = -vals  # Should go from 0 to 1
    n_pts = vals.shape[0]
    if xcoords is None:
        xcoords = range(n_pts)
    pts = np.vstack([xcoords, vals]).T

    last_vec = pts[-1] - pts[0]  # Get vec b/n first and last points
    last_vec_norm = last_vec / np.linalg.norm(last_vec)  # L2 normalized vector
    all_vecs = pts - pts[0]  # Move to origin at Point 1

    scalars = np.dot(all_vecs, last_vec_norm)  # Scalar of projection to last vector
    projections = np.outer(
        scalars, last_vec_norm
    )  # Projection of original vectors onto last vector

    # Get point w/ max difference b/n projection to diagonal and the original vector
    # This occurs at the knee point (similar to ROC graphs)
    vecs = all_vecs - projections
    dists = np.linalg.norm(vecs, axis=1)
    idx = np.argmax(dists)

    return idx


def nearest_neighbors(
    adata,
    n_neighbors: int,
    use_rep: str = 'X_pca',
    key_added: str = None,
    **kwargs,
):
    # Run kNN with RAPIDS
    X = cp.array(adata.obsm[use_rep])
    model = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X)
    distances, indices = model.kneighbors(X)

    # Index params for sparse matrix
    indptr = np.arange(0, indices.size+1, indices.shape[1])
    indices = indices.get()

    # Set similar to scanpy
    if key_added is not None:
        neighbors_key = key_added
        prefix = f'{key_added}_'
    else:
        neighbors_key = 'neighbors'
        prefix = ''

    adata.obsp[f'{prefix}connectivities'] = sp.sparse.csr_matrix((
        np.ones_like(indices).flatten(),
        indices.flatten(),
        indptr,
    ))

    adata.obsp[f'{prefix}distances'] = sp.sparse.csr_matrix((
        distances.get().flatten(),
        indices.flatten(),
        indptr,
    ))

    adata.uns[neighbors_key] = dict(
        connectivities_key=f'{prefix}connectivities',
        distances_key=f'{prefix}distances',
        params=dict(
            distance='euclidean',
            n_neighbors=n_neighbors,
            use_rep=use_rep
        ),
    )


def preprocess_rapids(
    adata,
    filter_min_counts: int = None,
    pca_layer: str = 'norm',
    pca_total_var: float = None,
    knn_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_n_epochs: int = 1000,
    phenograph_resolution: float = 1,
    umap_kwargs: dict = None,
):
    with tqdm(total=6) as pbar:
        # Filtering
        pbar.set_description("Filtering")
        adata.X = adata.raw.X.copy()
        if filter_min_counts is None:
            n_counts = np.sort(np.array(adata.X.sum(1)).flatten())
            idx = kneepoint(np.log10(n_counts))
            filter_min_counts = min(max(n_counts[idx], 5), 30)
        sc.pp.filter_cells(adata, min_counts=filter_min_counts)
        pbar.update(1)

        # Median library-size normalization
        pbar.set_description("Normalization")
        adata.layers['norm'] = adata.raw.X.copy()
        sc.pp.normalize_total(adata, layer='norm')
        
        # Log-transformation (natural log, pseudocount of 1)
        adata.layers['lognorm'] = adata.layers['norm'].copy()
        if 'log1p' in adata.uns:
            del adata.uns['log1p']
        sc.pp.log1p(adata, layer='lognorm')
        pbar.update(1)
        
        # Run PCA using GPU
        pbar.set_description("PCA")
        counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata.layers[pca_layer])
        model = cuml.PCA(n_components=min(adata.shape))
        X_pca = model.fit_transform(counts_sparse_gpu).get()
        cumulative_var = model.explained_variance_ratio_.get().cumsum()
        if pca_total_var is None:
            n_pcs = kneepoint(cumulative_var)
        else:
            n_pcs = np.argmin(abs(cumulative_var - pca_total_var))
        adata.obsm['X_pca'] = X_pca[:, :n_pcs]
        pbar.update(1)

        # kNN on PCA
        pbar.set_description("kNN")
        nearest_neighbors(adata, knn_neighbors, use_rep='X_pca')
        pbar.update(1)

        # Run UMAP using GPU
        with HiddenPrints():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                
                pbar.set_description("UMAP")
                if umap_kwargs is None:
                    umap_kwargs = dict()
                model = cuml.UMAP(
                    min_dist=umap_min_dist,
                    n_epochs=umap_n_epochs,
                    n_neighbors=knn_neighbors,
                    **umap_kwargs,
                )
                model.fit(X=X_pca[:, :n_pcs])
                X_umap = model.transform(X_pca[:, :n_pcs])
                adata.obsm['X_umap'] = X_umap
                pbar.update(1)

                # Cluster with GPU
                pbar.set_description("Clustering")
                min_size = -1
                kwargs = {'resolution': phenograph_resolution}
                phenograph_rapids(adata, min_size=min_size, **kwargs)
                
                pbar.update(1)
                pbar.set_description("Done")
