"""parquet_to_performance_compare.py
==================================
Batch wrapper around ``parquet_to_performance_pipeline`` that runs the
benchmark for **multiple segmentation columns** (e.g. Segger vs Cell‑Boundaries)
and aggregates the resulting summary metrics into comparison bar plots.

Public entry point
------------------
``parquet_to_performance_pipeline_compare``

Example
-------
>>> from pathlib import Path
>>> from sg_utils.pipelines.parquet_to_performance_compare import (
...     parquet_to_performance_pipeline_compare,
... )
>>> combined = parquet_to_performance_pipeline_compare(
...     parquet_path=Path("/data/my_transcripts.parquet"),
...     reference_h5ad=Path("/data/reference.h5ad"),
...     save_dir=Path("./compare_output"),
...     seg_cols=["segger_cell_id", "cell_boundaries_id", "nuclear_boundaries_id"],
... )
>>> combined.head()
                 Method                       Metric   Value
0       segger_cell_id           Number of Cells  107493
1  cell_boundaries_id           Number of Cells  126279
...
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sg_utils.pipelines.parquet_to_performance_from_mapping import parquet_to_performance_pipeline_from_mapping

__all__ = ["parquet_to_performance_pipeline_compare"]

LOGGER = logging.getLogger("parquet_pipeline_compare")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Main compare wrapper
# ---------------------------------------------------------------------------

def parquet_to_performance_pipeline_compare_from_mapping(
    *,
    parquet_path: str | Path,
    reference_h5ad: str | Path,
    save_dir: str | Path,
    seg_cols: Iterable[str],
    min_scores: Iterable[float],
    # The following kwargs are forwarded to the per‑method pipeline
    score_col: str | None = None,
    **pipeline_kwargs,
) -> pd.DataFrame:
    """Run performance benchmarking for several segmentation columns.

    Parameters
    ----------
    parquet_path
        Parquet transcript file.
    reference_h5ad
        Single‑cell reference atlas with ``.obs['cell_type']``.
    save_dir
        Root output directory. A sub‑folder is created for each *seg_col*.
    seg_cols
        Iterable of segmentation‑ID column names to evaluate.
    score_col
        Optional name of a confidence score column (used by the underlying
        pipeline via ``score_col`` kwarg). If *None*, the pipeline will infer
        it from *seg_col* internally.
    **pipeline_kwargs
        Additional keyword arguments forwarded to
        :func:`parquet_to_performance_pipeline` (e.g. filtering params).

    Returns
    -------
    pandas.DataFrame
        Long‑format table with columns ``[Method, Metric, Value]`` summarising
        all runs.
    """
    parquet_path = Path(parquet_path)
    reference_h5ad = Path(reference_h5ad)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []

    for index, seg in enumerate(seg_cols):
        LOGGER.info("\n―― Running pipeline for seg_col='%s' ――", seg)
        subdir = save_dir / seg
        summary_df = parquet_to_performance_pipeline_from_mapping(
            parquet_path=parquet_path,
            reference_h5ad=reference_h5ad,
            save_dir=subdir,
            seg_col=seg,
            score_col=score_col,
            overwrite=True,
            min_score=min_scores[index],
            **pipeline_kwargs,
        )
        # reshape to long format and append method column
        for _, row in summary_df.iterrows():
            all_records.append({"Method": seg, "Metric": row.Metric, "Value": row.Value})

    combined_df = pd.DataFrame(all_records)
    combined_df.to_csv(save_dir / "cell_typist_combined_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Comparison bar plots (one figure per metric)
    # ------------------------------------------------------------------
    sns.set_style("whitegrid")
    for metric, df_metric in combined_df.groupby("Metric"):
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_metric, x="Method", y="Value")
        plt.title(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"compare_{metric.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(save_dir / fname)
        plt.close()
        LOGGER.info("Saved comparison plot: %s", fname)

    LOGGER.info("All comparisons complete ✔")
    return combined_df
