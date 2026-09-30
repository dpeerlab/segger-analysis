"""Reference-based cell typing and marker genes.

Every segmentation is typed with the same parameter-free rule, and marker genes
are discovered once per reference before any segmentation is scored (Methods,
"Cell-type assignment and marker definitions").
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.special import entr

_INVALID_LABELS = {"", "nan", "none", "-1", "unknown"}


def _labels(ref: AnnData, groupby: str) -> tuple[np.ndarray, list[str]]:
    """Cell-type labels as strings and the sorted valid types; unlabelled cells belong to no type."""
    if not ref.var_names.is_unique:
        raise ValueError("reference gene names are not unique; call ref.var_names_make_unique() first")
    labels = ref.obs[groupby].astype(str).to_numpy()
    types = sorted({t for t in labels if t.lower() not in _INVALID_LABELS})
    return labels, types


def _counts(ref: AnnData, layer: str | None) -> sparse.csr_matrix:
    X = ref.layers[layer] if layer is not None else ref.X
    return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X, dtype=np.float64))


def reference_profiles(
    ref: AnnData,
    groupby: str,
    genes: list[str] | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """Mean raw expression of each reference cell type.

    This is the type profile used for cosine typing (Methods) and as the
    marker weights of positive marker recall.

    Parameters
    ----------
    ref
        Single-cell reference with raw counts.
    groupby
        Cell-type column in ``ref.obs``.
    genes
        Genes to keep, e.g. the iST panel. Genes absent from ``ref`` are dropped.
    layer
        Layer with raw counts; ``ref.X`` by default.

    Returns
    -------
    Types x genes table.
    """
    labels, types = _labels(ref, groupby)
    var = pd.Index(ref.var_names.astype(str))
    genes = list(var) if genes is None else [g for g in dict.fromkeys(genes) if g in var]
    X = _counts(ref, layer)[:, var.get_indexer(genes)]
    profiles = np.zeros((len(types), len(genes)))
    for i, t in enumerate(types):
        profiles[i] = np.asarray(X[labels == t].mean(axis=0)).ravel()
    return pd.DataFrame(profiles, index=types, columns=genes)


def assign_cell_types(
    adata: AnnData,
    profiles: pd.DataFrame,
    key_added: str = "cell_type",
) -> None:
    """Type cells by cosine similarity to reference profiles.

    The host type of a cell is the reference type with the highest cosine
    similarity over the genes shared between panel and reference. Confidence is
    ``1 - H(p) / log K`` with ``p_k ∝ max(cos_k, 0)²`` over the ``K`` types.

    Adds ``adata.obs[key_added]`` and ``adata.obs[key_added + "_confidence"]``.
    Cells without counts on the shared genes are left untyped.
    """
    genes = [g for g in profiles.columns if g in adata.var_names]
    X = adata[:, genes].X
    X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
    P = profiles[genes].to_numpy(dtype=np.float64)

    p_norm = np.linalg.norm(P, axis=1)
    p_norm[p_norm <= 0] = 1.0
    x_norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    has_counts = x_norm > 0
    x_norm[~has_counts] = 1.0
    cos = np.asarray(X @ P.T, dtype=np.float64) / x_norm[:, None] / p_norm[None, :]

    mass = np.clip(cos, 0.0, None) ** 2
    total = mass.sum(axis=1, keepdims=True)
    prob = np.divide(mass, total, out=np.full_like(mass, 1.0 / mass.shape[1]), where=total > 0)
    confidence = 1.0 - entr(prob).sum(axis=1) / np.log(prob.shape[1]) if prob.shape[1] > 1 else np.ones(len(prob))

    host = np.asarray(profiles.index)[cos.argmax(axis=1)].astype(object)
    host[~has_counts] = None
    adata.obs[key_added] = pd.Categorical(host, categories=list(profiles.index))
    adata.obs[f"{key_added}_confidence"] = np.where(has_counts, confidence, np.nan)


def rank_reference_genes(
    ref: AnnData,
    groupby: str,
    genes: list[str],
    layer: str | None = None,
) -> dict[str, pd.DataFrame]:
    """One-versus-rest Wilcoxon rank-sum test for every reference type.

    The reference is restricted to ``genes`` (the panel) and to labelled cells,
    normalised to 10,000 counts per cell and log-transformed; detection rates are
    taken on raw counts.

    Returns
    -------
    For every type, the scanpy result table with an extra ``detection`` column,
    the fraction of that type's cells with at least one count.
    """
    labels, types = _labels(ref, groupby)
    labelled = np.isin(labels, types)
    labels = labels[labelled]
    genes = [g for g in dict.fromkeys(genes) if g in ref.var_names]
    adata = ref[labelled, genes].copy()
    raw = _counts(adata, layer)
    adata.X = raw.copy()
    adata.obs[groupby] = pd.Categorical(labels, categories=types)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some cells have zero counts")
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby=groupby, groups=types, method="wilcoxon", use_raw=False)

    ranked = {}
    for t in types:
        df = sc.get.rank_genes_groups_df(adata, group=t)
        detection = np.asarray((raw[labels == t] > 0).mean(axis=0)).ravel()
        df["detection"] = df["names"].map(dict(zip(genes, detection)))
        ranked[t] = df
    return ranked


def positive_markers(
    ranked: dict[str, pd.DataFrame],
    n: int = 12,
    max_pval_adj: float = 0.01,
    min_log2fc: float = 1.0,
    min_detection: float = 0.5,
) -> dict[str, list[str]]:
    """Positive markers of every type: significant, enriched and detected in most of its cells.

    Genes with adjusted P below ``max_pval_adj``, log2 fold change above
    ``min_log2fc`` and detection in at least ``min_detection`` of the type's cells;
    the ``n`` highest-scoring are kept.
    """
    markers = {}
    for t, df in ranked.items():
        keep = (
            (df["pvals_adj"] < max_pval_adj)
            & (df["logfoldchanges"] > min_log2fc)
            & (df["detection"] >= min_detection)
        )
        markers[t] = df[keep].sort_values("scores", ascending=False).head(n)["names"].tolist()
    return markers


def contamination_markers(
    ranked: dict[str, pd.DataFrame],
    n: int = 12,
    max_pval_adj: float = 0.01,
    min_log2fc: float = 1.0,
    max_detection: float = 0.1,
) -> dict[str, list[str]]:
    """Contamination markers of every type: significantly depleted and nearly absent.

    Genes with adjusted P below ``max_pval_adj``, log2 fold change below
    ``-min_log2fc`` and detection in fewer than ``max_detection`` of the type's
    cells; the ``n`` most depleted are kept. Transcripts of these genes in a cell
    of that type are counted as false positives.
    """
    markers = {}
    for t, df in ranked.items():
        keep = (
            (df["pvals_adj"] < max_pval_adj)
            & (df["logfoldchanges"] < -min_log2fc)
            & (df["detection"] < max_detection)
        )
        markers[t] = df[keep].sort_values("scores", ascending=True).head(n)["names"].tolist()
    return markers


def celltypist_labels(adata: AnnData, model: str, chunk_size: int = 100_000) -> pd.Series:
    """Predict cell types with a CellTypist model (Domínguez Conde et al., 2022).

    Raw counts are normalised to 100 per cell and log-transformed, as for model
    training; each cell gets the best-scoring type (no majority voting), so cells
    are predicted in chunks of ``chunk_size`` to bound memory.
    """
    level = logging.getLogger().level
    import celltypist

    logging.getLogger().setLevel(level)  # celltypist switches root logging to INFO on import
    clf = celltypist.Model.load(str(model))
    genes = [g for g in clf.features if g in adata.var_names]
    # CellTypist warns unless counts are normalised to 10,000; these models use 100.
    ct_log = logging.getLogger("celltypist")
    ct_level = ct_log.level
    ct_log.setLevel(logging.ERROR)
    labels = []
    try:
        for start in range(0, adata.n_obs, chunk_size):
            chunk = adata[start : start + chunk_size, genes]
            query = AnnData(X=chunk.X.copy(), obs=pd.DataFrame(index=chunk.obs_names), var=pd.DataFrame(index=genes))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some cells have zero counts")
                sc.pp.normalize_total(query, target_sum=100)
            sc.pp.log1p(query)
            predictions = celltypist.annotate(query, model=clf, majority_voting=False)
            labels.append(predictions.predicted_labels["predicted_labels"].astype(str))
    finally:
        ct_log.setLevel(ct_level)
    return pd.concat(labels)


def detection_rates(
    ref: AnnData,
    groupby: str,
    genes: list[str],
    layer: str | None = None,
) -> pd.DataFrame:
    """Fraction of each type's cells with at least one count of each gene (types x genes)."""
    labels, types = _labels(ref, groupby)
    genes = [g for g in dict.fromkeys(genes) if g in ref.var_names]
    X = _counts(ref[:, genes], layer)
    rates = np.vstack([np.asarray((X[labels == t] > 0).mean(axis=0)).ravel() for t in types])
    return pd.DataFrame(rates, index=types, columns=genes)
