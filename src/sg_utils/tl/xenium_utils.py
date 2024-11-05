import pandas as pd
import scanpy as sc
import numpy as np
import scipy as sp


def anndata_from_transcripts(
    transcripts: pd.DataFrame,
    cell_label: str,
    gene_label: str,
):
    # Feature names to indices
    ids_cell, labels_cell = pd.factorize(transcripts[cell_label])
    ids_gene, labels_gene = pd.factorize(transcripts[gene_label])
    
    # Remove NaN values
    mask = ids_cell >= 0
    ids_cell = ids_cell[mask]
    ids_gene = ids_gene[mask]
    
    # Sort row index
    order = np.argsort(ids_cell)
    ids_cell = ids_cell[order]
    ids_gene = ids_gene[order]
    
    # Build sparse matrix
    X = sp.sparse.coo_matrix(
        (
            np.ones_like(ids_cell),
            np.stack([ids_cell, ids_gene]),
        ),
        shape=(len(labels_cell), len(labels_gene)),
    ).tocsr()

    # To AnnData
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=labels_cell.astype(str)),
        var=pd.DataFrame(index=labels_gene),
    )
    adata.raw = adata.copy()
    
    return adata