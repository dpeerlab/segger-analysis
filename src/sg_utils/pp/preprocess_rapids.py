from sg_utils.tl.phenograph_rapids import phenograph_rapids
from tqdm import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import cupy as cp
import warnings
import cupyx
import cuml
import sys
import os


# Class to suppress output from inside functions (e.g., PhenoGraph)
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def preprocess_rapids(
    adata,
    filter_min_counts: int = 50,
    pca_layer: str = 'norm',
    pca_total_var: float = 0.5,
    knn_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    phenograph_resolution: float = 1,
):
    with tqdm(total=6) as pbar:
        # Filtering
        pbar.set_description("Filtering")
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
        n_pcs = np.argmin(abs(cumulative_var - pca_total_var))
        adata.obsm['X_pca'] = X_pca[:, :n_pcs]
        pbar.update(1)

        # kNN on PCA
        pbar.set_description("kNN")
        X_pca = cp.array(adata.obsm['X_pca'])
        model = cuml.neighbors.NearestNeighbors(n_neighbors=knn_neighbors)
        model.fit(X_pca)
        distances, indices = model.kneighbors(X_pca)
        pbar.update(1)

        # Run UMAP using GPU
        with HiddenPrints():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                
                pbar.set_description("UMAP")
                model = cuml.UMAP(
                    min_dist=umap_min_dist,
                    n_epochs=1000,
                    n_neighbors=knn_neighbors,
                )
                model.fit(X=X_pca[:, :n_pcs])
                X_umap = model.transform(X_pca[:, :n_pcs])
                adata.obsm['X_umap'] = X_umap.get()
                pbar.update(1)

                # Cluster with GPU
                pbar.set_description("Clustering")
                min_size = -1
                kwargs = {'resolution': phenograph_resolution}
                clusters = phenograph_rapids(indices, min_size=min_size, kwargs=kwargs)
                adata.obs['phenograph_cluster'] = pd.Categorical(clusters)
                pbar.update(1)
                pbar.set_description("Done")
