import numpy as np
import anndata as ad
from sklearn.metrics import calinski_harabasz_score, silhouette_score, f1_score
from typing import Dict, List, Tuple
from itertools import combinations
import pandas as pd
import scanpy as sc


def compute_clustering_scores(
    adata: ad.AnnData, cell_type_column: str = "celltype_major", use_pca: bool = True
) -> Tuple[float, float]:
    """
    Compute the Calinski-Harabasz and Silhouette scores for clustering in an AnnData object.

    Parameters:
    ----------
    adata : AnnData
        Annotated data object with clustering data.
    cell_type_column : str, default="celltype_major"
        Column in `adata.obs` containing cell type annotations.
    use_pca : bool, default=True
        Whether to use PCA-transformed features or raw expression data.

    Returns:
    -------
    Tuple[float, float]
        Calinski-Harabasz score and Silhouette score.
    """
    if cell_type_column not in adata.obs:
        raise ValueError(f"Column '{cell_type_column}' must be present in adata.obs.")

    # Sample up to 10,000 cells for efficiency
    n_cells = min(adata.n_obs, 10_000)
    cell_indices = np.random.choice(adata.n_obs, n_cells, replace=False)

    # Select features
    features = adata.obsm["X_pca"] if use_pca else adata.X
    features = features[cell_indices, :]
    labels = adata.obs[cell_type_column].iloc[cell_indices].values

    # Compute scores
    ch_score = calinski_harabasz_score(features, labels)
    sh_score = silhouette_score(features, labels)

    return ch_score, sh_score


def find_markers(
    adata: ad.AnnData,
    cell_type_column: str,
    pos_percentile: float = 5,
    neg_percentile: float = 10,
    percentage: float = 50,
    min_cells: int = 10
) -> Dict[str, Dict[str, List[str]]]:
    """
    Identify positive and negative markers for each cell type based on gene expression.

    Parameters:
    ----------
    adata : AnnData
        Annotated data object containing gene expression data.
    cell_type_column : str
        Column name in `adata.obs` specifying cell types.
    pos_percentile : float, default=5
        Top x% of highly expressed genes to consider as markers.
    neg_percentile : float, default=10
        Bottom x% of lowly expressed genes to consider.
    percentage : float, default=50
        Minimum percentage of cells within a type that must express a marker.

    Returns:
    -------
    Dict[str, Dict[str, List[str]]]
        Dictionary mapping cell types to lists of positive and negative marker genes.
    """
    markers = {}
    adata.raw = adata  # Ensure raw expression values are used
    adata.var_names_make_unique()
    
    # if the cell type does not meet minimum cell requirements remove it
    for cell_type in adata.obs[cell_type_column].unique():
        if adata.obs[cell_type_column].value_counts()[cell_type] < min_cells:
            print(f"Removing {cell_type} due to insufficient cells.")
            adata = adata[adata.obs[cell_type_column] != cell_type]

    sc.tl.rank_genes_groups(adata, groupby=cell_type_column)
    
    genes = adata.var_names
    for cell_type in adata.obs[cell_type_column].unique():
        subset = adata[adata.obs[cell_type_column] == cell_type]
        mean_expression = np.asarray(subset.X.mean(axis=0)).flatten()

        # Compute percentile cutoffs
        cutoff_high = np.percentile(mean_expression, 100 - pos_percentile)
        cutoff_low = np.percentile(mean_expression, neg_percentile)

        pos_indices = np.where(mean_expression >= cutoff_high)[0]
        neg_indices = np.where(mean_expression <= cutoff_low)[0]

        # Filter positive markers based on expression percentage
        expr_frac = np.asarray((subset.X[:, pos_indices] > 0).mean(axis=0)).flatten()
        valid_pos_indices = pos_indices[expr_frac >= (percentage / 100)]

        markers[cell_type] = {
            "positive": genes[valid_pos_indices].tolist(),
            "negative": genes[neg_indices].tolist(),
        }

    return markers


def find_mutually_exclusive_genes(
    adata: ad.AnnData, markers: Dict[str, Dict[str, List[str]]], cell_type_column: str
) -> List[Tuple[str, str]]:
    """
    Identify mutually exclusive genes based on expression criteria.

    Parameters:
    ----------
    adata : AnnData
        Annotated data object containing gene expression data.
    markers : Dict[str, Dict[str, List[str]]]
        Dictionary of positive and negative markers per cell type.
    cell_type_column : str
        Column name in `adata.obs` specifying cell types.

    Returns:
    -------
    List[Tuple[str, str]]
        List of mutually exclusive gene pairs.
    """
    exclusive_genes = {}
    all_exclusive = []
    gene_expression = adata.to_df()

    for cell_type, marker_sets in markers.items():
        positive_markers = marker_sets["positive"]
        exclusive_genes[cell_type] = []

        for gene in positive_markers:
            gene_expr = gene_expression[gene]
            cell_type_mask = adata.obs[cell_type_column] == cell_type
            non_cell_type_mask = ~cell_type_mask

            # Define mutually exclusive criteria
            if (gene_expr[cell_type_mask] > 0).mean() > 0.2 and (gene_expr[non_cell_type_mask] > 0).mean() < 0.05:
                exclusive_genes[cell_type].append(gene)
                all_exclusive.append(gene)

    unique_genes = list(set(all_exclusive))
    filtered_exclusive_genes = {
        cell_type: [gene for gene in exclusive_genes[cell_type] if gene in unique_genes]
        for cell_type in exclusive_genes.keys()
    }

    return [
        (gene1, gene2)
        for key1, key2 in combinations(filtered_exclusive_genes.keys(), 2)
        for gene1 in filtered_exclusive_genes[key1]
        for gene2 in filtered_exclusive_genes[key2]
    ]


def compute_MECR(adata: ad.AnnData, gene_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """
    Compute the Mutually Exclusive Co-expression Rate (MECR) for each gene pair.

    Parameters:
    ----------
    adata : AnnData
        Annotated data object containing gene expression data.
    gene_pairs : List[Tuple[str, str]]
        List of gene pairs to evaluate.

    Returns:
    -------
    Dict[Tuple[str, str], float]
        Dictionary mapping gene pairs to their MECR values.
        Returns np.nan for pairs where one or both genes are not found in adata.
    """    
    mecr_dict = {}
    try:
        gene_expression_df = adata.to_df()        
    except Exception as e:                
        for gene1, gene2 in gene_pairs:
            mecr_dict[(gene1, gene2)] = np.nan
        return mecr_dict

    pair_counter = 0
    for gene1, gene2 in gene_pairs:
        pair_counter += 1

        # Check if genes exist in the DataFrame's columns (i.e., were in adata.var_names)
        if gene1 not in gene_expression_df.columns:            
            mecr_dict[(gene1, gene2)] = np.nan
            continue  # Skip to the next pair
        if gene2 not in gene_expression_df.columns:            
            mecr_dict[(gene1, gene2)] = np.nan
            continue  # Skip to the next pair

        # Extract binarized expression (True if expression > 0, False otherwise)
        expr_gene1_series = gene_expression_df[gene1] # Raw expression Series for gene1
        expr_gene2_series = gene_expression_df[gene2] # Raw expression Series for gene2

        expr_gene1_bool = expr_gene1_series > 0  # Boolean Series for gene1
        expr_gene2_bool = expr_gene2_series > 0  # Boolean Series for gene2
        
        both_expressed_proportion = (expr_gene1_bool & expr_gene2_bool).mean()
        at_least_one_expressed_proportion = (expr_gene1_bool | expr_gene2_bool).mean()
        
        mecr_value = 0.0 # Default value as per original logic if at_least_one_expressed_proportion is 0
        if at_least_one_expressed_proportion > 0:
            mecr_value = both_expressed_proportion / at_least_one_expressed_proportion            
        mecr_dict[(gene1, gene2)] = mecr_value

    return mecr_dict


def calculate_sensitivity(
    adata: ad.AnnData,
    purified_markers: Dict[str, Dict[str, List[str]]],
    max_cells_per_type: int = 1000,
    cell_type_column: str = "celltype_major",
    random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """
    Calculate sensitivity of purified markers for each cell type.

    Parameters:
    ----------
    adata : AnnData
        Annotated data object containing gene expression data.
    purified_markers : Dict[str, Dict[str, List[str]]]
        Dictionary mapping cell types to positive and negative markers.
    max_cells_per_type : int, default=1000
        Maximum number of cells to consider per cell type.
    cell_type_column : str, default="celltype_major"
        Column in adata.obs containing cell type annotations.

    Returns:
    -------
    Dict[str, List[float]]
        Sensitivity values for each cell type.
    """
    rng = np.random.default_rng(random_state)
    sensitivity_results = {cell_type: [] for cell_type in purified_markers.keys()}

    for cell_type, markers in purified_markers.items():
        positive_markers = markers.get("positive", [])

        subset = adata[adata.obs[cell_type_column] == cell_type]
        if subset.n_obs == 0:
            continue

        if subset.n_obs > max_cells_per_type:
            cell_indices = rng.choice(subset.n_obs, max_cells_per_type, replace=False)
            subset = subset[cell_indices]

        positive_indices = subset.var_names.get_indexer(positive_markers)
        missing_count = (positive_indices == -1).sum()
        if verbose: print(f"  {missing_count} markers not found in subset.var_names")

        # Filter out invalid indices
        valid_indices = positive_indices[positive_indices >= 0]
        if len(valid_indices) == 0:
            if verbose: print("  No valid marker genes found in this subset. Skipping.")
            continue
        for i, cell_counts in enumerate(subset.X):
            total_counts = cell_counts.sum()

            try:
                positive_counts = cell_counts[:, valid_indices].sum()
            except IndexError as e:
                if verbose: print(f"  [ERROR] IndexError at cell {i}: {e}")
                continue

            sensitivity = positive_counts / total_counts if total_counts > 0 else 0
            sensitivity_results[cell_type].append(sensitivity)

        if verbose: print(f"  Finished {cell_type}. Avg sensitivity: {np.mean(sensitivity_results[cell_type]):.3f}")
    return sensitivity_results