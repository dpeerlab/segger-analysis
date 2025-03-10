from typing import List, Optional
import pandas as pd
import scanpy as sc
import numpy as np
import scipy as sp
import anndata as ad

def anndata_from_transcripts(
    transcripts: pd.DataFrame,
    cell_label: str,
    gene_label: str,
    coordinate_labels: List[str] = None,
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

    # Add spatial coords
    if coordinate_labels is not None:
        coords = transcripts[coordinate_labels]
        centroids = coords.groupby(transcripts[cell_label]).mean()
        idx = adata.obs.index.astype(transcripts[cell_label].dtype)
        adata.obsm['X_spatial'] = centroids.loc[idx].values
    
    return adata





def filter_transcripts(
    transcripts_df: pd.DataFrame,
    min_qv: float = 30.0,
) -> pd.DataFrame:
    """
    Filters transcripts based on quality value and removes unwanted transcripts.

    Parameters:
        transcripts_df (pd.DataFrame): The dataframe containing transcript data.
        min_qv (float): The minimum quality value threshold for filtering transcripts.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    filter_codewords = (
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword_",
        "BLANK_",
        "DeprecatedCodeword_",
        "UnassignedCodeword_",
    )
    mask = transcripts_df["qv"].ge(min_qv)
    transcripts_df["feature_name"] = transcripts_df["feature_name"].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    mask &= ~transcripts_df["feature_name"].str.startswith(filter_codewords)
    return transcripts_df[mask]


def calculate_cell_summary(group, min_transcripts, cell_id_col, convex_hull=True):
    if len(group) < min_transcripts:
        return None
    
    if convex_hull:
        cell_hull = ConvexHull(group[["x_location", "y_location"]])
    else:
        cell_hull = generate_boundary(group[["x_location", "y_location"]])
    cell_area = cell_hull.area

    return {
        "cell": group[cell_id_col].iloc[0],
        "cell_centroid_x": group["x_location"].mean(),
        "cell_centroid_y": group["y_location"].mean(),
        "cell_area": cell_area,
    }

# Define a wrapper function to replace lambda
def process_cell_group(args):
    return calculate_cell_summary(*args)



def create_anndata(
    df: pd.DataFrame,                 
    primary_df: pd.DataFrame,         
    panel_df: Optional[pd.DataFrame] = None,
    min_transcripts: int = 5,
    max_transcripts: int = 2000,
    cell_id_col: str = "cell_id",
    qv_threshold: float = 30,
    min_cell_area: float = 10.0,
    max_cell_area: float = 1000.0,
    num_workers: int = 4  
) -> ad.AnnData:
    """
    Create an AnnData object from a transcriptomics dataset.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing transcript-level data with at least 'cell_id' and 'feature_name' columns.
    primary_df : pd.DataFrame
        Original DataFrame with all transcripts before filtering.
    panel_df : Optional[pd.DataFrame], default=None
        DataFrame containing metadata for genes.
    min_transcripts : int, default=5
        Minimum transcript count threshold per cell.
    max_transcripts : int, default=2000
        Maximum transcript count threshold per cell.
    cell_id_col : str, default="cell_id"
        Column name for cell identifiers.
    qv_threshold : float, default=30
        Quality value threshold for filtering transcripts.
    min_cell_area : float, default=10.0
        Minimum area threshold for cells.
    max_cell_area : float, default=1000.0
        Maximum area threshold for cells.
    num_workers : int, default=4
        Number of parallel workers for processing.

    Returns:
    -------
    ad.AnnData
        An AnnData object containing processed single-cell transcriptomics data.
    """

    # Filter out invalid cell IDs
    filtered_df = df.dropna(subset=[cell_id_col])

    # Group by cell and gene, count occurrences
    grouped_counts = (
        filtered_df.rename(columns={cell_id_col: "cell", "feature_name": "gene"})
        .groupby(["cell", "gene"]).size().reset_index(name="count")
    )

    # Create a pivot table (cell-gene matrix)
    pivot_table = (
        grouped_counts.pivot(index="cell", columns="gene", values="count")
        .fillna(0).astype("int16")
    )

    # Filter cells based on transcript counts
    total_counts = pivot_table.sum(axis=1)
    filtered_pivot = pivot_table[(total_counts >= min_transcripts) & (total_counts <= max_transcripts)]

    # Compute global transcript statistics
    global_total_transcripts = primary_df.shape[0]
    assigned_transcripts = filtered_df.shape[0]
    global_percent_assigned = (assigned_transcripts / global_total_transcripts) * 100

    # Compute %assigned for each gene
    gene_total_counts = primary_df.groupby("feature_name").size()
    gene_assigned_counts = filtered_df.groupby("feature_name").size()
    gene_percent_assigned = (gene_assigned_counts / gene_total_counts * 100).fillna(0)

    # Extract cell and gene names
    cell_indices = filtered_pivot.index.astype(str).tolist()
    gene_names = filtered_pivot.columns.values

    # Convert to sparse matrix
    X = sp.csr_matrix(filtered_pivot.values)

    # Create AnnData object
    adata = ad.AnnData(X=X, dtype="int16")
    adata.obs_names = cell_indices  
    adata.var_names = gene_names    

    # Add transcript count metadata
    adata.obs["transcripts"] = X.sum(axis=1).A1  
    adata.obs["unique_transcripts"] = (X > 0).sum(axis=1).A1  

    # Prepare cell groups for parallel processing
    pqdm_args = [(group, min_transcripts, cell_id_col) for _, group in filtered_df.groupby(cell_id_col)]

    # Run parallel processing to compute cell boundaries
    cell_summary_results = pqdm(
        pqdm_args, 
        process_cell_group,  
        n_jobs=num_workers, 
        desc="Computing cell boundaries",  
        leave=True  
    )

    # Remove None results and convert to DataFrame
    cell_summary_df = pd.DataFrame(
        [summary for summary in cell_summary_results if summary is not None]
    ).set_index("cell")

    # Ensure alignment of cell indices
    valid_cells = list(set(cell_summary_df.index) & set(cell_indices))
    adata = adata[valid_cells, :]
    cell_summary_df.index = [str(x) for x in cell_summary_df.index]

    # Merge additional metadata into AnnData
    adata.obs = adata.obs.merge(
        cell_summary_df.loc[valid_cells], left_index=True, right_index=True
    )

    # Add gene metadata
    if panel_df is not None:
        adata.var = panel_df.set_index("gene")
    else:
        adata.var = pd.DataFrame(
            [{"gene": gene, "feature_types": "Gene Expression", "genome": "Unknown"}
             for gene in gene_names]
        ).set_index("gene")

    # Add %assigned for each gene
    adata.var["percent_assigned"] = gene_percent_assigned.reindex(adata.var.index).fillna(0).values

    # Store global transcript assignment percentage
    adata.uns["prop_assigned"] = global_percent_assigned

    return adata




def preprocess_adata(
    reference_adata: ad.AnnData, query_adata: ad.AnnData, transfer_column: str, min_counts: int = 5, max_counts: int = 500
) -> ad.AnnData:
    """Annotate query AnnData object using a scRNA-seq reference atlas.

    Args:
    - reference_adata: ad.AnnData
        Reference AnnData object containing the scRNA-seq atlas data.
    - query_adata: ad.AnnData
        Query AnnData object containing the data to be annotated.
    - transfer_column: str
        The name of the column in the reference atlas's `obs` to transfer to the query dataset.

    Returns:
    - query_adata: ad.AnnData
        Annotated query AnnData object with transferred labels and UMAP coordinates from the reference.
    """
    common_genes = list(set(reference_adata.var_names) & set(query_adata.var_names))
    reference_adata = reference_adata[:, common_genes]
    query_adata = query_adata[:, common_genes]
    sc.pp.filter_cells(query_adata, min_counts=min_counts)
    # sc.pp.filter_cells(query_adata, max_counts=max_counts)
    query_adata.layers["raw"] = query_adata.raw.X if query_adata.raw else query_adata.X
    query_adata.var["raw_counts"] = query_adata.layers["raw"].sum(axis=0)
    sc.pp.normalize_total(query_adata, target_sum=1e4)
    sc.pp.log1p(query_adata)
    sc.tl.ingest(query_adata, reference_adata, obs=transfer_column)
    query_adata.obsm["X_umap_ref"] = query_adata.obsm["X_umap"]
    sc.pp.pca(query_adata)
    sc.pp.neighbors(query_adata)
    sc.tl.umap(query_adata)
    return query_adata




