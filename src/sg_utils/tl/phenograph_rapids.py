import cudf
import cugraph
import cupy
import numpy as np
import pandas as pd


def phenograph_rapids(
    adata,
    neighbors_key: str = 'neighbors',
    key_added: str = 'phenograph_cluster',
    min_size: int = -1,
    **kwargs,
):
    # Get indices from AnnData
    row, col = adata.obsp[adata.uns[neighbors_key]['connectivities_key']].nonzero()
    splits = np.unique(row, return_index=True)[1][1:]
    indices = cupy.array(np.split(col, splits))

    # Run phenograph
    clusters = _run_phenograph(indices, min_size, **kwargs)
    adata.obs[key_added] = pd.Categorical(clusters)


def _run_phenograph(
    indices: cupy.ndarray,
    min_size: int = -1,
    **kwargs,
):
    # Build kNN graph in GPU
    n, k = indices.shape
    edges = cudf.concat([
        cudf.Series(np.repeat(np.arange(n), k), name='source'), # sources
        cudf.Series(indices.flatten(), name='destination'), # targets
    ], axis=1)
    G = cugraph.from_cudf_edgelist(edges)
    
    # Build jaccard-weighted graph in GPU
    jaccard_edges = cugraph.jaccard(G, edges[['source', 'destination']])
    G = cugraph.from_cudf_edgelist(jaccard_edges, *jaccard_edges.columns)
    
    # Cluster jaccard-weighted graph
    result, score = cugraph.louvain(G, **kwargs)
    
    # Sort clusters by size
    sizes = result['partition'].value_counts()
    sizes.loc[:] = cupy.where(sizes > min_size, cupy.arange(len(sizes)), -1)
    result['partition'] = result['partition'].map(sizes)
    
    # Sort by vertex (e.g. cell)
    clusters = result.sort_values('vertex')['partition'].values.get()
    
    return clusters
