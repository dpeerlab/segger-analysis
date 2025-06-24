"""
parquet_to_performance.py
=========================
Reusable CosMx/Xenium benchmarking pipeline.

This module exposes a single public entry point – :func:`parquet_to_performance_pipeline` – that
wraps the full workflow previously implemented in *cosmx_pipeline_v2.py*.

The pipeline performs the following high‑level steps:

1. **Load & quality‑filter** a parquet transcript file (codeword/QV filters, segger score filter).
2. **Convert** the filtered transcripts into an :class:`scanpy.AnnData` object.
3. **Preprocess** with RAPIDS‑accelerated utilities (library‑size filtering, PCA, UMAP, Phenograph).
4. **Harmonise genes** between the query and a reference single‑cell atlas (h5ad).
5. **Train/Load** a *CellTypist* model on the reference (subsampling for class balance).
6. **Annotate** query cells via CellTypist, with majority‑vote overclustering.
7. **Compute performance metrics** (median sensitivity, MECR) and summary statistics.
8. **Generate quick diagnostic plots** (UMAPs, per‑metric bar charts).
9. **Persist** outputs (summary CSVs, AnnData, plots) under a user‑specified output directory.

The function returns the summary :class:`pandas.DataFrame` so the caller can programmatically use
the results.

Example
-------
>>> from pathlib import Path
>>> from sg_utils.pipelines.parquet_to_performance import parquet_to_performance_pipeline
>>> summary = parquet_to_performance_pipeline(
...     parquet_path=Path("/data/merscope_brain.parquet"),
...     reference_h5ad=Path("/data/SEA_AD_DLPFC_brain_atlas_subset_100k.h5ad"),
...     save_dir=Path("./brain_pipeline_output"),
... )
>>> summary
                    Metric        Value
0          Number of Cells  107493.0000
1  Average Cell Area (µm²)      -1.0000
2  Median Counts per Cell    304.0000
3  Overall Median Sensitivity    0.1511
4      Overall Median MECR    0.0451
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mygene  # type: ignore
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.spatial import QhullError

import celltypist as ct  # type: ignore

sc.settings.random_state = 42
# Project‑local helpers
from sg_utils.tl.xenium_utils import anndata_from_transcripts
from sg_utils.pp.preprocess_rapids import preprocess_rapids
from sg_utils.tl.phenograph_rapids import phenograph_rapids
from sg_utils.tl.comparison_metrics import (
    find_markers,
    calculate_sensitivity,
    find_mutually_exclusive_genes,
    compute_MECR,
)

__all__ = ["parquet_to_performance_pipeline"]

LOGGER = logging.getLogger("parquet_pipeline")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

###################################################################################
# Helper functions (kept private – not exported)                                  #
###################################################################################

def _filter_transcripts(df: pd.DataFrame, min_qv: float = 30.0, min_score: float =0.0, score_label="score", gene_label = "feature_name") -> pd.DataFrame:
    """Remove low‑quality or control probe transcripts."""
    filter_codewords = (
        "NegControlProbe_",
        "antisense_",
        "NegControlCodeword_",
        "BLANK_",
        "DeprecatedCodeword_",
        "UnassignedCodeword_",
    )
    if "qv" not in df.columns:
        logging.warning(
            "No 'qv' column found in the DataFrame – skipping quality filtering."
        )
        mask = ~df[gene_label].str.startswith(filter_codewords)
    else:
         mask = df["qv"].ge(min_qv) & ~df[gene_label].str.startswith(filter_codewords)
    result = df.loc[mask]
    if score_label in df.columns:
        print(f"Filtering transcripts with {score_label} >= {min_score}")
        result = result[result[score_label] >= min_score]
    if result.empty:
        raise ValueError(
            f"No transcripts remaining after filtering with qv >= {min_qv} and {score_label} >= {min_score}."
        )
    return result


def _load_and_filter_transcripts(
    parquet_path: Path,
    seg_col: str,
    n_sample: int | None = None,
    min_qv: float = 30.0,
    min_score: float = 0.0,
    score_label: str = "score",
    gene_label: str = "feature_name",
) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(parquet_path)
    df = _filter_transcripts(df, min_qv=min_qv, gene_label=gene_label, score_label=score_label, min_score=min_score)

    if n_sample is not None:
        df = df.sample(n=n_sample, random_state=42)
    return df


def _build_anndata(
    df: pd.DataFrame,
    seg_col: str,
    coords: Tuple[str, str],
    gene_label: str,
):
    return anndata_from_transcripts(
        df, cell_label=seg_col, gene_label=gene_label, coordinate_labels=list(coords)
    )


def _harmonise_genes(query: sc.AnnData, reference: sc.AnnData):
    mg = mygene.MyGeneInfo()
    ens_clean = reference.var_names.str.replace(r"\.\d+$", "", regex=True)
    res = mg.querymany(
        ens_clean.tolist(), scopes="ensembl.gene", fields="symbol", species="human", verbose=False
    )
    id2sym = {
        r["query"]: (r.get("symbol") or r["query"]).upper()
        for r in res
        if isinstance(r, dict) and "query" in r
    }

    ens_series = pd.Series(ens_clean.values, index=ens_clean)
    reference.var["gene_symbol"] = ens_series.map(id2sym).fillna(ens_series).str.upper()
    reference.var_names = reference.var["gene_symbol"]
    reference.var_names_make_unique()

    query.var_names = query.var_names.str.upper(); query.var_names_make_unique()

    shared = sorted(set(query.var_names) & set(reference.var_names))
    if len(shared) < 10:
        LOGGER.warning("Only %d shared genes – results may be unstable.", len(shared))
    return query[:, shared].copy(), reference[:, shared].copy()


def _save_umap(ad: sc.AnnData, color, fname: Path, title: str | None = None):
    """
    Saves a UMAP plot, ensuring the legend and title are not cut off.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    sc.pl.umap(
        ad,
        color=color,
        ax=ax,
        show=False,
        legend_loc='right margin',  # avoids legend clutter
        legend_fontsize='small',    # helps if many keys
        title=title,                # scanpy handles title better here
    )
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)

###################################################################################
# Public pipeline                                                                 #
###################################################################################


def parquet_to_performance_pipeline(
    *,
    parquet_path: str | Path,
    reference_h5ad: str | Path,
    save_dir: str | Path,
    # Identification columns
    seg_col: str = "cell_boundaries_id",
    score_col: str = "score",
    min_score: float = 0.0,
    coords: Tuple[str, str] = ("global_x", "global_y"),
    gene_label: str = "feature_name",
    # Sampling & geometry
    n_sample: int | None = None,
    # Pre‑processing hyper‑parameters
    filter_min_counts: int = 5,
    pca_total_var: float = 0.75,
    umap_min_dist: float = 0.25,
    umap_n_epochs: int = 4000,
    knn_neighbors: int = 20,
    phenograph_res: float = 1.0,
    # CellTypist parameters
    ct_subsample_per_type: int = 2000,
    pos_percentile: int = 90,
    neg_percentile: int = 10,
    pct_expressed: int = 50,
    max_cells_per_type: int = 2000,
    ct_iterations: int = 100,
    ct_log: int = 100,
    # Misc
    overwrite: bool = True,
    verbose: bool = False,
    # For plot names
    sample_name: str | None = "Sample",  # Used in plot titles and file names
) -> pd.DataFrame:
    """Run the full CosMx/Xenium benchmark and return summary metrics.

    Parameters
    ----------
    parquet_path
        Path to *xxx_transcripts.parquet*.
    reference_h5ad
        Single‑cell atlas with ``.obs['cell_type']`` annotation.
    save_dir
        Directory where all artefacts will be written (created if absent).
    seg_col, coords, gene_label
        Column names in the parquet for segmentation IDs, spatial coordinates, and gene label.
    n_sample
        If provided, randomly subsample *n_sample* transcripts for quick debugging.
    pixel_um
        Conversion factor from pixel area to µm² for cell area estimation.
    filter_min_counts, pca_total_var, …
        Pre‑processing hyper‑parameters mirroring *preprocess_rapids* & downstream analysis.
    ct_subsample_per_type
        Number of cells per cell‑type to sample when training the CellTypist model.
    overwrite
        If *False* and *save_dir* exists, the function will raise instead of overwriting outputs.

    Returns
    -------
    pandas.DataFrame
        A 5‑row table of summary performance metrics (same as the original script).
    """
    parquet_path = Path(parquet_path)
    reference_h5ad = Path(reference_h5ad)
    save_dir = Path(save_dir)

    if save_dir.exists() and not overwrite:
        raise FileExistsError(f"Save directory '{save_dir}' already exists – set overwrite=True to replace.")
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s", save_dir)

    #  Visualise Segger confidence score distribution
    raw_df = pd.read_parquet(parquet_path)
    if score_col in raw_df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(raw_df[score_col].dropna(), bins=50)
        plt.xlabel("Segger score")
        plt.ylabel("Count")
        plt.title(f"Distribution of {sample_name} Segger confidence scores")
        plt.tight_layout()
        plt.savefig(save_dir / f"segger_score_distribution.png")
        plt.close()
        LOGGER.info("Segger score distribution plot saved.")
    else:
        LOGGER.warning("Segger score column '%s' not found – skipping histogram.", score_col)
    
    # Parameter checks
    # Make sure seg_col and coords exist in the DataFrame
    if not all(col in raw_df.columns for col in [seg_col, *coords, gene_label]):
        raise ValueError(
            f"Required columns '{seg_col}', {coords}, or '{gene_label}' not found in the parquet file."
        )
    # 1. Load → filter transcripts → AnnData
    df = _load_and_filter_transcripts(parquet_path, seg_col, n_sample, gene_label=gene_label, score_label=score_col, min_score=min_score)

    adata = _build_anndata(df, seg_col, coords, gene_label)


    # 2. RAPIDS preprocessing & Phenograph
    preprocess_rapids(
        adata,
        filter_min_counts=filter_min_counts,
        pca_total_var=pca_total_var,
        umap_min_dist=umap_min_dist,
        umap_n_epochs=umap_n_epochs,
        pca_layer="lognorm",
        knn_neighbors=knn_neighbors,
        phenograph_resolution=phenograph_res,
    )

    # Raw count metrics
    adata.obs["raw_count"] = adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else np.ravel(adata.X.sum(axis=1))
    adata.obs[["raw_count"]].to_csv(save_dir / "transcripts_per_cell.csv", index=True)
    median_counts = float(np.median(adata.obs["raw_count"]))
    LOGGER.info("Cells: %d | Median counts: %.1f", adata.n_obs, median_counts)

    # 3. Load reference & harmonise genes
    scRNAseq = sc.read_h5ad(reference_h5ad)
    adata, scRNAseq = _harmonise_genes(adata, scRNAseq)
    adata.raw = adata; scRNAseq.raw = scRNAseq

    # 4. CellTypist model training
    #   – sample for class balance then train a fresh model (fast for <50k cells)
    sample_idx = (
        scRNAseq.obs.groupby("cell_type").sample(ct_subsample_per_type, replace=True, random_state=42).index.drop_duplicates()
    )
    scRNAseq.layers["norm_100"] = scRNAseq.X.copy()
    sc.pp.normalize_total(scRNAseq, layer="norm_100", target_sum=ct_log)
    scRNAseq.layers["lognorm_100"] = scRNAseq.layers["norm_100"].copy()
    if "log1p" in scRNAseq.uns:
        del scRNAseq.uns["log1p"]
    sc.pp.log1p(scRNAseq, layer="lognorm_100")

    ct_model = ct.train(
        scRNAseq[sample_idx],
        labels="cell_type",
        check_expression=False,
        n_jobs=os.cpu_count() or 32,
        max_iter=ct_iterations,
        random_state=42,
    )

    # 5. Query normalisation (100 UMI per cell; log‑transform)
    adata.layers["norm_100"] = adata.raw.X.copy()
    sc.pp.normalize_total(adata, layer="norm_100", target_sum=ct_log)
    adata.layers["lognorm_100"] = adata.layers["norm_100"].copy()
    if "log1p" in adata.uns:
        del adata.uns["log1p"]
    sc.pp.log1p(adata, layer="lognorm_100")
    adata.X = adata.layers["lognorm_100"]

    # 6. Cell annotation
    preds = ct.annotate(
        adata,
        model=ct_model,
        majority_voting=True,
        over_clustering="phenograph_cluster",
        min_prop=0.2,
    )
    adata.obs["celltypist_label"] = preds.predicted_labels["predicted_labels"]
    adata.obs["celltypist_label_cluster"] = preds.predicted_labels["majority_voting"]
    adata.obs["celltypist_probability"] = preds.probability_matrix.max(1)

    # 7. UMAPs
    if "X_umap" not in scRNAseq.obsm:
        sc.pp.neighbors(scRNAseq, n_neighbors=knn_neighbors)
        sc.tl.umap(scRNAseq, min_dist=umap_min_dist)
    _save_umap(scRNAseq, "cell_type", save_dir / "reference_umap.png", f"{sample_name} Reference atlas")

    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, min_dist=umap_min_dist)
    _save_umap(adata, ["celltypist_label"], save_dir / f"{sample_name}_umap.png", f"{sample_name} UMAP")
    
    # fuse adata and scRNAseq for plotting
    adata_for_merge = adata.copy()
    adata_for_merge.obs["cell_type"] = adata_for_merge.obs["celltypist_label"]
    merged_adata = adata_for_merge.concatenate(scRNAseq, batch_key="dataset", batch_categories=["query", "reference"], index_unique="-")
    _save_umap(merged_adata, ["cell_type"], save_dir / f"{sample_name}_merged_umap.png", f"{sample_name} Merged UMAP")

    # 8. Marker‑based metrics
    markers = find_markers(
        scRNAseq,
        "cell_type",
        neg_percentile=neg_percentile,
        percentage=pct_expressed,
        pos_percentile=pos_percentile,
    )
    sens = calculate_sensitivity(adata, markers, max_cells_per_type, "celltypist_label")

    flattened_sens_values = [s for v in sens.values() for s in v]
    # Create a DataFrame from the flattened values
    flattened_sens_df = pd.DataFrame({"Sensitivity": flattened_sens_values})
    # Save the flattened DataFrame to a CSV
    flattened_sens_df.to_csv(save_dir / "all_gene_sensitivities_flattened.csv", index=False)

    excl = find_mutually_exclusive_genes(scRNAseq, markers, "cell_type")
    mecr_raw = compute_MECR(adata, excl)

    overall_sens = float(np.nanmedian([s for v in sens.values() for s in v]))
    overall_mecr = float(np.nanmedian(list(mecr_raw.values())))

    summary = pd.DataFrame(
        {
            "Metric": [
                "Number of Cells",
                "Median Counts per Cell",
                "Overall Median Sensitivity",
                "Overall Median MECR",
            ],
            "Value": [adata.n_obs, median_counts, overall_sens, overall_mecr],
        }
    )
    summary.to_csv(save_dir / "summary_metrics.csv", index=False)

    # 9. Quick bar plots
    for _, row in summary.iterrows():
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=[row.Metric], y=[row.Value], ax=ax)
        ax.set_ylabel(row.Metric)
        plt.tight_layout()
        plt.title(f"{sample_name} {row.Metric}")
        plt.savefig(save_dir / f"plot_{row.Metric.lower().replace(' ', '_')}.png")
        plt.close()

    # Sensitivity & MECR per cell type
    sens_df = pd.DataFrame({"Cell Type": sens.keys(), "Median Sensitivity": [np.nanmedian(v) for v in sens.values()]})
    sens_df.to_csv(save_dir / "sensitivity_by_celltype.csv", index=False)
    mecr_df = pd.DataFrame({"Cell Type": mecr_raw.keys(), "MECR": mecr_raw.values()})
    mecr_df.to_csv(save_dir / "mecr_by_celltype.csv", index=False)

    # Persist AnnData
    adata.write(save_dir / "final_adata.h5ad", compression="gzip")

    # tidy up
    gc.collect()
    return summary
