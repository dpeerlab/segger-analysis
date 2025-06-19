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
                       Metric      Value
0             Number of Cells  107493.0000
1   Average Cell Area (µm²)      -1.0000
2    Median Counts per Cell     304.0000
3  Overall Median Sensitivity      0.1511
4   Overall Median MECR      0.0451
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
# Project-local helpers
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

    # Optional Segger score filter (if present)
    if "segger" in seg_col.lower():
        score_col = seg_col.replace("segger_cell_id", "score")
        if score_col in df.columns:
            df = df[df[score_col] > 0.75]
        else:
            LOGGER.warning(
                "Segger score column '%s' not found – skipping score filter.", score_col
            )

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
    sc.pl.umap(ad, color=color, ax=ax, show=False, legend_loc='on data')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

def _map_cosmx_to_segger(
    transcripts_df: pd.DataFrame,
    seg_col: str,
    cosmx_id_col: str,
    annotations_df: pd.DataFrame,
    external_cell_id_col: str,
) -> pd.DataFrame:
    """Map CosMX cell IDs to Segger IDs and join with annotations."""
    LOGGER.info(f"Mapping {cosmx_id_col} to {seg_col}")
    if cosmx_id_col not in transcripts_df.columns:
        raise ValueError(f"CosMX ID column '{cosmx_id_col}' not found in transcript dataframe.")

    grouping_cols = list(dict.fromkeys([seg_col, cosmx_id_col]))
    mapping_df = transcripts_df.groupby(grouping_cols).size().reset_index(name='counts')
    best_match = mapping_df.loc[mapping_df.groupby(seg_col)['counts'].idxmax()].copy()

    # FIX: Ensure merge keys are the same type (string) to prevent ValueError.
    best_match.loc[:, cosmx_id_col] = best_match[cosmx_id_col].astype(str)
    annotations_df.loc[:, external_cell_id_col] = annotations_df[external_cell_id_col].astype(str)

    # Merge with external annotations
    annotated_mapping = best_match.merge(
        annotations_df, left_on=cosmx_id_col, right_on=external_cell_id_col, how="inner"
    )
    LOGGER.info(f"Successfully mapped {len(annotated_mapping)} {seg_col} entities to external annotations.")
    return annotated_mapping

###################################################################################
# Public pipeline                                                                 #
###################################################################################

def parquet_to_performance_pipeline_from_mapping(
    *,
    parquet_path: str | Path,
    reference_h5ad: str | Path,
    save_dir: str | Path,
    # External annotation parameters
    external_annotations_csv: str | Path | None = None,
    cosmx_id_col: str = "cell",
    nucleus_seg_col: str = "nucleus_boundaries_id",
    external_cell_id_col: str = "cell_ID",
    external_annotation_label: str = "cell_types",
    # Identification columns
    seg_col: str = "segger_cell_id",
    score_col: str = "score",
    min_score: float = 0.0,
    coords: Tuple[str, str] = ("global_x", "global_y"),
    gene_label: str = "feature_name",
    # Sampling & geometry
    n_sample: int | None = None,
    # Pre-processing hyper-parameters
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
        Single-cell atlas with ``.obs['cell_type']`` annotation. Used for default pipeline.
    save_dir
        Directory where all artefacts will be written (created if absent).
    external_annotations_csv
        Path to a CSV file with external cell type annotations. If provided, triggers new workflows.
    cosmx_id_col
        Column in the parquet file corresponding to the cell IDs in the external annotation CSV.
    nucleus_seg_col
        Column in the parquet file for nucleus segmentation IDs.
    external_cell_id_col, external_annotation_label
        Column names in the external CSV for cell IDs and their labels.
    seg_col, coords, gene_label
        Column names in the parquet for segmentation IDs, spatial coordinates, and gene label.
    ... (other parameters) ...
    overwrite
        If *False* and *save_dir* exists, the function will raise instead of overwriting outputs.

    Returns
    -------
    pandas.DataFrame
        A table of summary performance metrics.
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
    required_cols = [seg_col, *coords, gene_label]
    if external_annotations_csv:
        required_cols.extend([cosmx_id_col, nucleus_seg_col])
    if not all(col in raw_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in raw_df.columns]
        raise ValueError(f"Required columns {missing} not found in the parquet file.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~ BRANCH 1: EXTERNAL ANNOTATIONS PROVIDED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if external_annotations_csv:
        external_annotations_csv = Path(external_annotations_csv)
        if not external_annotations_csv.exists():
            raise FileNotFoundError(f"External annotations file not found: {external_annotations_csv}")

        annots_df = pd.read_csv(external_annotations_csv)
        for col in [external_cell_id_col, external_annotation_label]:
            if col not in annots_df.columns:
                raise ValueError(f"Column '{col}' not found in external annotations CSV.")

        # --- Setup ID Mappings ---
        nucleus_to_type_df = _map_cosmx_to_segger(raw_df, nucleus_seg_col, cosmx_id_col, annots_df, external_cell_id_col)
        segger_to_type_df = _map_cosmx_to_segger(raw_df, seg_col, cosmx_id_col, annots_df, external_cell_id_col)

        # --- SCENARIO 1: DIRECT ANNOTATION & METRICS ---
        LOGGER.info("\n--- Running Scenario 1: Direct Annotation Metrics ---")
        s1_save_dir = save_dir / "direct_annotation_metrics"
        s1_save_dir.mkdir(exist_ok=True)

        # Build nucleus AnnData to define markers
        adata_nuc = _build_anndata(raw_df, nucleus_seg_col, coords, gene_label)
        adata_nuc.obs = adata_nuc.obs.merge(
            nucleus_to_type_df[[nucleus_seg_col, external_annotation_label]],
            left_index=True, right_on=nucleus_seg_col, how='left'
        ).set_index(adata_nuc.obs.index)
        adata_nuc_annotated = adata_nuc[adata_nuc.obs[external_annotation_label].notna()].copy()
        LOGGER.info(f"Found {adata_nuc_annotated.n_obs} annotated nuclei to define markers.")

        # Convert the annotation column to 'category' dtype for scanpy.
        adata_nuc_annotated.obs[external_annotation_label] = adata_nuc_annotated.obs[external_annotation_label].astype('category')

        # Preprocess nucleus data
        sc.pp.normalize_total(adata_nuc_annotated, target_sum=ct_log)
        sc.pp.log1p(adata_nuc_annotated)

        # pp for umap
        sc.pp.pca(adata_nuc_annotated)
        sc.pp.neighbors(adata_nuc_annotated)
        sc.tl.umap(adata_nuc_annotated)

        _save_umap(
            adata_nuc_annotated, [external_annotation_label],
            s1_save_dir / f"{sample_name}_nucleus_umap.png",
            f"{sample_name} Nucleus UMAP (Annotated)"
        )
        # Define markers from this ground-truth data
        markers = find_markers(
            adata_nuc_annotated, external_annotation_label,
            neg_percentile=neg_percentile, percentage=pct_expressed, pos_percentile=pos_percentile
        )
        excl = find_mutually_exclusive_genes(adata_nuc_annotated, markers, external_annotation_label)

        # Build main segmentation AnnData and calculate metrics
        df_filt = _filter_transcripts(raw_df, min_score=min_score, score_label=score_col, gene_label=gene_label)
        adata_main = _build_anndata(df_filt, seg_col, coords, gene_label)
        adata_main.obs = adata_main.obs.merge(
            segger_to_type_df[[seg_col, external_annotation_label]],
            left_index=True, right_on=seg_col, how='left'
        ).set_index(adata_main.obs.index)
        adata_main_annotated = adata_main[adata_main.obs[external_annotation_label].notna()].copy()

        sens = calculate_sensitivity(adata_main_annotated, markers, max_cells_per_type, external_annotation_label)
        mecr_raw = compute_MECR(adata_main_annotated, excl)
        overall_sens = float(np.nanmedian([s for v in sens.values() for s in v]))
        overall_mecr = float(np.nanmedian(list(mecr_raw.values())))
        median_counts = float(np.median(adata_main_annotated.X.sum(axis=1).A1))

        summary_s1 = pd.DataFrame({
            "Metric": ["Number of Cells", "Median Counts per Cell", "Overall Median Sensitivity", "Overall Median MECR"],
            "Value": [adata_main_annotated.n_obs, median_counts, overall_sens, overall_mecr],
        })
        summary_s1.to_csv(s1_save_dir / "summary_metrics.csv", index=False)
        LOGGER.info("Scenario 1 Summary:\n%s", summary_s1)

        # --- SCENARIO 2: RE-TRAIN CELLTYPIST & RE-ANNOTATE ---
        LOGGER.info("\n--- Running Scenario 2: Retrain CellTypist & Re-annotate ---")
        s2_save_dir = save_dir / "retrained_celltypist_metrics"
        s2_save_dir.mkdir(exist_ok=True)

        # Use annotated nucleus data from S1 for training
        adata_nuc_train = adata_nuc_annotated.copy()

        ct_model = ct.train(
            adata_nuc_train, labels=external_annotation_label, check_expression=False,
            n_jobs=os.cpu_count() or 32, max_iter=ct_iterations, random_state=42
        )
        LOGGER.info("New CellTypist model trained on provided annotations.")

        # Process the full main AnnData for prediction
        adata = _build_anndata(df_filt, seg_col, coords, gene_label)
        preprocess_rapids(
            adata, filter_min_counts=filter_min_counts, pca_total_var=pca_total_var,
            umap_min_dist=umap_min_dist, umap_n_epochs=umap_n_epochs, pca_layer="lognorm",
            knn_neighbors=knn_neighbors, phenograph_resolution=phenograph_res,
        )
        adata.obs["raw_count"] = adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else np.ravel(adata.X.sum(axis=1))
        adata.obs[["raw_count"]].to_csv(save_dir / "transcripts_per_cell.csv", index=True)
        # Normalize for prediction
        adata.layers["norm_100"] = adata.raw.X.copy()
        sc.pp.normalize_total(adata, layer="norm_100", target_sum=ct_log)
        adata.layers["lognorm_100"] = adata.layers["norm_100"].copy()
        if "log1p" in adata.uns: del adata.uns["log1p"]
        sc.pp.log1p(adata, layer="lognorm_100")
        adata.X = adata.layers["lognorm_100"]

        # Annotate with the new model
        preds = ct.annotate(
            adata, model=ct_model, majority_voting=True,
            over_clustering="phenograph_cluster", min_prop=0.2,
        )
        adata.obs["celltypist_label"] = preds.predicted_labels["predicted_labels"]
        adata.obs["celltypist_label_cluster"] = preds.predicted_labels["majority_voting"]

        # Calculate metrics using markers from S1
        sens_s2 = calculate_sensitivity(adata, markers, max_cells_per_type, "celltypist_label")
        mecr_raw_s2 = compute_MECR(adata, excl)
        overall_sens_s2 = float(np.nanmedian([s for v in sens_s2.values() for s in v]))
        overall_mecr_s2 = float(np.nanmedian(list(mecr_raw_s2.values())))
        median_counts_s2 = float(np.median(adata.raw.X.sum(axis=1).A1))
         

        summary_s2 = pd.DataFrame({
            "Metric": ["Number of Cells", "Median Counts per Cell", "Overall Median Sensitivity", "Overall Median MECR"],
            "Value": [adata.n_obs, median_counts_s2, overall_sens_s2, overall_mecr_s2],
        })
        summary_s2.to_csv(s2_save_dir / "summary_metrics.csv", index=False)
        LOGGER.info("Scenario 2 Summary:\n%s", summary_s2)

        # Generate plots and save artifacts for S2
        _save_umap(adata, ["celltypist_label_cluster"], s2_save_dir / f"{sample_name}_umap.png", f"{sample_name} UMAP (Retrained Model)")
        adata.write(s2_save_dir / "final_adata.h5ad", compression="gzip")
        # more stats to save
        sens_df = pd.DataFrame({"Cell Type": sens.keys(), "Median Sensitivity": [np.nanmedian(v) for v in sens.values()]})
        sens_df.to_csv(save_dir / "sensitivity_by_celltype.csv", index=False)
        mecr_df = pd.DataFrame({"Cell Type": mecr_raw.keys(), "MECR": mecr_raw.values()})
        mecr_df.to_csv(save_dir / "mecr_by_celltype.csv", index=False)

        final_summary = summary_s2

    # tidy up
    gc.collect()
    return final_summary