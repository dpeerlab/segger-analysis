"""Segmentation quality metrics (Methods, "Benchmark metrics" and "Colorectal
cancer contamination and recall").

Metrics are computed from each method's assigned transcripts, mostly through a
cell x gene count matrix, so every segmentation is scored by the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree

from ..io import _UNASSIGNED


def cell_by_gene(
    transcripts: pl.DataFrame,
    method: str,
    genes: list[str] | None = None,
    min_transcripts: int = 0,
) -> AnnData:
    """Cell x gene counts of one segmentation.

    Parameters
    ----------
    transcripts
        One row per transcript with ``feature_name``, ``x``, ``y`` and a column
        ``method`` holding the assigned cell (null when unassigned).
    method
        Column with the cell ids of the segmentation.
    genes
        Genes of the matrix, all detected genes by default. Transcripts of other
        genes are ignored, also for ``n_transcripts`` and ``min_transcripts``.
    min_transcripts
        Minimum number of assigned transcripts per cell.

    Returns
    -------
    AnnData with raw counts, ``obs["n_transcripts"]`` and the cell centroid (mean
    transcript position) in ``obsm["spatial"]``.
    """
    tx = transcripts.lazy().filter(pl.col(method).is_not_null())
    if genes is None:
        genes = sorted(tx.select(pl.col("feature_name").cast(pl.Utf8).unique()).collect().to_series())
    genes = list(dict.fromkeys(genes))
    tx = (
        tx.select(
            pl.col(method).alias("cell"),
            pl.col("feature_name").cast(pl.Enum(list(genes)), strict=False).to_physical().alias("j"),
            "x",
            "y",
        )
        .filter(pl.col("j").is_not_null())
        .collect()
    )
    cells = (
        tx.group_by("cell")
        .agg(pl.len().alias("n_transcripts"), pl.col("x").mean(), pl.col("y").mean())
        .filter(pl.col("n_transcripts") >= min_transcripts)
        .with_columns(pl.col("cell").cast(pl.Utf8).alias("name"))
        .sort("name")
        .with_row_index("i")
    )
    counts = tx.join(cells.select("cell", "i"), on="cell").group_by("i", "j").len()
    X = sparse.csr_matrix(
        (
            counts["len"].to_numpy().astype(np.float32),
            (counts["i"].to_numpy(), counts["j"].to_numpy()),
        ),
        shape=(cells.height, len(genes)),
    )
    obs = pd.DataFrame(
        {"n_transcripts": cells["n_transcripts"].to_numpy()},
        index=pd.Index(cells["name"].to_list(), name=method),
    )
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    adata.obsm["spatial"] = cells.select("x", "y").to_numpy()
    return adata


def coverage(transcripts: pl.DataFrame, method: str) -> float:
    """Fraction of transcripts assigned to a cell."""
    return float(transcripts.get_column(method).is_not_null().mean())


def positive_marker_recall(
    adata: AnnData,
    markers: dict[str, list[str]],
    profiles: pd.DataFrame,
    transcripts: pl.DataFrame,
    radius: float = 10.0,
    cell_type_key: str = "cell_type",
) -> pd.Series:
    """Per-cell positive marker recall (PMR, %).

    A positive marker of the host type is eligible for a cell when at least one
    transcript of that gene, assigned or not, lies within ``radius`` µm of the cell
    centroid. Eligible markers are weighted by their mean expression in the host
    type of the reference, and a marker is recovered when the cell holds at least
    one of its transcripts.

    Parameters
    ----------
    adata
        Output of :func:`cell_by_gene`, typed with
        :func:`sg_utils.tl.celltyping.assign_cell_types`.
    markers
        Positive markers per type (:func:`~sg_utils.tl.celltyping.positive_markers`).
    profiles
        Reference type profiles, used as marker weights.
    transcripts
        All transcripts of the section (``feature_name``, ``x``, ``y``); defines
        which markers are physically available around each cell.
    radius
        Vicinity radius in µm.

    Returns
    -------
    PMR per cell, NaN for cells without an eligible marker. A marker missing from
    ``adata`` is eligible but never recovered.
    """
    host = adata.obs[cell_type_key].astype(str).to_numpy()
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    X = adata.X.tocsc()
    var = {g: j for j, g in enumerate(adata.var_names)}
    marker_genes = sorted({g for genes in markers.values() for g in genes})
    missing = [g for g in marker_genes if g not in profiles.columns]
    if missing:
        raise KeyError(f"markers without a reference profile: {missing}")
    positions = (
        transcripts.lazy()
        .filter(pl.col("feature_name").is_in(marker_genes))
        .select(pl.col("feature_name").cast(pl.Utf8), "x", "y")
        .collect()
        .partition_by("feature_name", as_dict=True)
    )
    trees = {key[0]: cKDTree(df.select("x", "y").to_numpy()) for key, df in positions.items()}

    recovered = np.zeros(adata.n_obs)
    eligible = np.zeros(adata.n_obs)
    bound = np.nextafter(radius, np.inf)
    for cell_type, genes in markers.items():
        rows = np.flatnonzero(host == cell_type)
        if rows.size == 0:
            continue
        for g in genes:
            if g not in trees:
                continue
            dist, _ = trees[g].query(xy[rows], distance_upper_bound=bound)
            available = np.isfinite(dist)
            weight = float(profiles.loc[cell_type, g])
            present = np.asarray(X[rows, var[g]].toarray()).ravel() > 0 if g in var else np.zeros(rows.size, bool)
            eligible[rows] += weight * available
            recovered[rows] += weight * (available & present)
    with np.errstate(invalid="ignore", divide="ignore"):
        pmr = np.where(eligible > 0, 100.0 * recovered / eligible, np.nan)
    return pd.Series(pmr, index=adata.obs_names, name="pmr")


def exclusive_gene_pairs(
    transcripts: pl.DataFrame,
    max_jaccard: float = 0.10,
    n_pairs: int = 500,
    min_gene_count: int = 10,
    min_nucleus_fraction: float = 0.01,
) -> pd.DataFrame:
    """Gene pairs that are semi-exclusive in nuclei.

    Nuclear transcripts are the internal ground truth of spurious co-expression:
    two genes are semi-exclusive when the Jaccard index of the nuclei detecting
    them is below ``max_jaccard``. The ``n_pairs`` most exclusive pairs are kept.

    Parameters
    ----------
    transcripts
        All transcripts with the vendor ``cell_id`` and ``overlaps_nucleus``.
    max_jaccard
        Largest nuclear Jaccard index of a semi-exclusive pair.
    n_pairs
        Number of pairs kept, most exclusive first; ties keep the gene order.
    min_gene_count
        Minimum number of nuclear transcripts of a gene.
    min_nucleus_fraction
        Minimum fraction of nuclei (and at least three) detecting each gene of a pair.

    Returns
    -------
    One row per pair with its nuclear Jaccard index, the number of nuclei
    detecting each gene and each gene's cytoplasmic-to-nuclear count ratio. The
    genes used to discover pairs are stored in ``attrs["genes"]``.
    """
    tx = transcripts.filter(pl.col("cell_id").is_not_null() & ~pl.col("cell_id").cast(pl.Utf8).is_in(_UNASSIGNED))
    nuclear = tx.filter(pl.col("overlaps_nucleus") == 1)
    gene_counts = nuclear.group_by("feature_name").len().filter(pl.col("len") >= min_gene_count)
    genes = sorted(gene_counts.get_column("feature_name").cast(pl.Utf8).to_list())
    code = pl.col("feature_name").cast(pl.Enum(genes), strict=False).to_physical().alias("j")
    nuclei = nuclear.select("cell_id").unique().with_row_index("i")
    presence = (
        nuclear.select("cell_id", code)
        .drop_nulls("j")
        .join(nuclei, on="cell_id")
        .select("i", "j")
        .unique()
    )
    B = sparse.csr_matrix(
        (np.ones(presence.height), (presence["i"].to_numpy(), presence["j"].to_numpy())),
        shape=(nuclei.height, len(genes)),
    )
    occurrence = np.asarray(B.sum(axis=0)).ravel()
    co = (B.T @ B).toarray()
    jaccard = co / (occurrence[:, None] + occurrence[None, :] - co + 1e-12)
    np.fill_diagonal(jaccard, 1.0)

    frequent = occurrence >= max(3, int(nuclei.height * min_nucleus_fraction))
    mask = (jaccard < max_jaccard) & frequent[:, None] & frequent[None, :]
    i, j = np.where(np.triu(mask, 1))
    order = np.argsort(jaccard[i, j], kind="stable")[:n_pairs]
    i, j = i[order], j[order]

    compartment = tx.select(code, (pl.col("overlaps_nucleus") == 1).alias("nuclear")).drop_nulls("j")
    counts = compartment.group_by("j").agg(pl.col("nuclear").sum().alias("n_nuc"), (~pl.col("nuclear")).sum().alias("n_cyto"))
    n_nuc = np.zeros(len(genes))
    n_cyto = np.zeros(len(genes))
    n_nuc[counts["j"].to_numpy()] = counts["n_nuc"].to_numpy()
    n_cyto[counts["j"].to_numpy()] = counts["n_cyto"].to_numpy()
    ratios = n_cyto / np.maximum(n_nuc, 1)
    names = np.asarray(genes, dtype=object)
    pairs = pd.DataFrame(
        {
            "gene_a": names[i],
            "gene_b": names[j],
            "nuclear_jaccard": jaccard[i, j],
            "nuclei_a": occurrence[i],
            "nuclei_b": occurrence[j],
            "cyto_ratio_a": ratios[i],
            "cyto_ratio_b": ratios[j],
        }
    )
    pairs.attrs["genes"] = genes
    return pairs


def spurious_coexpression(
    adata: AnnData,
    pairs: pd.DataFrame,
    min_transcripts: int = 20,
    genes: list[str] | None = None,
) -> pd.Series:
    """Spurious co-expression (SCE) of semi-exclusive gene pairs; lower is better.

    Counts are L1-normalised per cell and the co-expression of a pair is the soft
    Jaccard index ``Σ min / Σ max`` over cells. The score is the mean excess over
    the nuclear Jaccard index, weighted by how often both genes occur in nuclei
    and down-weighted for cytoplasmic genes, ``1 / sqrt((1 + r_a)(1 + r_b))``.

    Parameters
    ----------
    adata
        Output of :func:`cell_by_gene`.
    pairs
        Output of :func:`exclusive_gene_pairs`.
    genes
        Genes that count towards ``min_transcripts`` and the normalisation, by
        default those used for pair discovery (``pairs.attrs["genes"]``).

    Returns
    -------
    ``sce``, its approximate 95% confidence half-width ``ci95`` and ``n_pairs``;
    NaN when no cell or no pair can be scored.
    """
    if genes is None:
        genes = pairs.attrs.get("genes", list(adata.var_names))
    genes = [g for g in genes if g in adata.var_names]
    X = adata[:, genes].X.tocsr().astype(np.float64)
    totals = np.asarray(X.sum(axis=1)).ravel()
    X = X[totals >= min_transcripts]
    totals = totals[totals >= min_transcripts]
    col = {g: k for k, g in enumerate(genes)}
    ok = pairs["gene_a"].isin(col) & pairs["gene_b"].isin(col)
    p = pairs[ok]
    if X.shape[0] == 0 or p.empty:
        return pd.Series({"sce": np.nan, "ci95": np.nan, "n_pairs": len(p)})
    X = sparse.diags(1.0 / totals) @ X
    Xa = X[:, p["gene_a"].map(col).to_numpy()]
    Xb = X[:, p["gene_b"].map(col).to_numpy()]
    soft_jaccard = np.asarray(Xa.minimum(Xb).sum(axis=0)).ravel() / (
        np.asarray(Xa.maximum(Xb).sum(axis=0)).ravel() + 1e-12
    )
    excess = np.maximum(soft_jaccard - p["nuclear_jaccard"].to_numpy(), 0.0)
    weights = np.sqrt(p["nuclei_a"] * p["nuclei_b"]).to_numpy() / np.sqrt(
        (1.0 + p["cyto_ratio_a"]) * (1.0 + p["cyto_ratio_b"])
    ).to_numpy()

    sce = float(np.average(excess, weights=weights))
    n_eff = weights.sum() ** 2 / np.square(weights).sum()
    variance = float(np.average((excess - sce) ** 2, weights=weights))
    return pd.Series({"sce": sce, "ci95": 1.96 * np.sqrt(variance / n_eff), "n_pairs": len(p)})


def pair_host_genes(
    pairs: pd.DataFrame,
    profiles: pd.DataFrame,
    detection: pd.DataFrame,
    max_detection: float = 0.01,
) -> dict[str, list[str]]:
    """Host genes of the cross-type semi-exclusive pairs, per cell type.

    Each gene is assigned to the reference type with its highest mean expression.
    Pairs are visited from most to least exclusive; for a pair of genes from two
    different types, each gene is the host gene of its own type with the other as
    its contamination partner. Every (type, contamination partner) combination
    keeps only the host gene of its most exclusive pair, and only partners detected
    in fewer than ``max_detection`` of that type's reference cells are used.
    Transcripts of host genes assigned to cells of the matching type are the
    "clean" transcripts of :func:`specificity`.
    """
    genes = [g for g in dict.fromkeys(pairs["gene_a"].tolist() + pairs["gene_b"].tolist()) if g in profiles]
    sub = profiles[genes]
    gene_type = sub.idxmax(axis=0).where(sub.max(axis=0) > 0)

    partner_of: dict[tuple[str, str], str] = {}
    for a, b in pairs.sort_values("nuclear_jaccard", kind="stable")[["gene_a", "gene_b"]].itertuples(index=False):
        ta, tb = gene_type.get(a), gene_type.get(b)
        if pd.isna(ta) or pd.isna(tb) or ta == tb:
            continue
        partner_of.setdefault((ta, b), a)
        partner_of.setdefault((tb, a), b)

    host: dict[str, set[str]] = {}
    for (cell_type, contaminant), gene in partner_of.items():
        if contaminant in detection and detection.at[cell_type, contaminant] < max_detection:
            host.setdefault(cell_type, set()).add(gene)
    return {t: sorted(g) for t, g in host.items()}


def marker_counts(
    adata: AnnData,
    genes_by_type: dict[str, list[str]],
    cell_type_key: str = "cell_type",
) -> np.ndarray:
    """Per-cell number of transcripts from the gene set of the cell's own type."""
    host = adata.obs[cell_type_key].astype(str).to_numpy()
    var = {g: j for j, g in enumerate(adata.var_names)}
    X = adata.X.tocsr()
    out = np.zeros(adata.n_obs)
    for cell_type, genes in genes_by_type.items():
        rows = np.flatnonzero(host == cell_type)
        cols = [var[g] for g in genes if g in var]
        if rows.size and cols:
            out[rows] = np.asarray(X[rows][:, cols].sum(axis=1)).ravel()
    return out


def specificity(
    adata: AnnData,
    host_genes: dict[str, list[str]],
    contamination_genes: dict[str, list[str]],
    min_transcripts: int = 20,
    cell_type_key: str = "cell_type",
    n_boot: int = 600,
    seed: int = 0,
) -> pd.Series:
    """Specificity, clean / (clean + FP), pooled over transcripts.

    Clean transcripts are host genes (:func:`pair_host_genes`) assigned to cells of
    their type; false positives (FP) are contamination markers
    (:func:`~sg_utils.tl.celltyping.contamination_markers`) of the cell's type.
    Cells with at least ``min_transcripts`` transcripts and marker evidence are
    pooled; the 95% confidence interval comes from resampling cells.

    Returns
    -------
    ``specificity``, ``ci_low``, ``ci_high``, ``n_cells``, ``clean`` and ``fp``.
    """
    clean = marker_counts(adata, host_genes, cell_type_key)
    fp = marker_counts(adata, contamination_genes, cell_type_key)
    keep = (adata.obs["n_transcripts"].to_numpy() >= min_transcripts) & (clean + fp > 0)
    clean, evidence = clean[keep], (clean + fp)[keep]

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, clean.size, clean.size)
        boot[b] = clean[idx].sum() / evidence[idx].sum()
    return pd.Series(
        {
            "specificity": clean.sum() / evidence.sum(),
            "ci_low": np.percentile(boot, 2.5),
            "ci_high": np.percentile(boot, 97.5),
            "n_cells": int(keep.sum()),
            "clean": clean.sum(),
            "fp": evidence.sum() - clean.sum(),
        }
    )
