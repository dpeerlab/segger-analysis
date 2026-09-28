"""Whole-transcriptome sections with about 10^9 transcripts (Atera).

A section of this size does not fit in memory, so every pass reads
``transcripts.parquet`` and the Segger output in blocks of rows joined on
``row_index`` and keeps only per-cell or per-gene sums. The metrics are those of the
Atera comparison of Segger and 10x Cell (Supplementary Fig. 10): cell counts and
coverage on the whole section; cell typing, positive marker recall (PMR),
cell-typing confidence and separation on a shared subset of cells; spurious
co-expression against a nuclear reference.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from scipy import sparse

# Curated cervical markers: 17 cell types, 115 genes, each marking one type.
MARKERS = {
    "Epithelial_Squamous": ["KRT5", "KRT14", "KRT6A", "KRT17", "KRT15", "TP63", "KRT13", "KRT4", "SPRR1B", "CRNN", "SFN", "EPCAM"],
    "Epithelial_Glandular": ["KRT8", "KRT18", "KRT19", "MUC5B", "MUC5AC", "WFDC2", "CEACAM5", "CLDN3"],
    "Tcell": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "IL7R", "CD8A", "CD7"],
    "Treg": ["FOXP3", "CTLA4", "IL2RA", "TNFRSF4", "IKZF2"],
    "NK": ["GNLY", "NKG7", "KLRD1", "KLRF1", "NCR1", "KLRB1"],
    "Bcell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "TCL1A"],
    "Plasma": ["MZB1", "IGHG1", "IGKC", "JCHAIN", "XBP1", "DERL3", "TNFRSF17", "SDC1"],
    "Myeloid": ["CD68", "CD163", "LYZ", "CD14", "C1QA", "C1QB", "C1QC", "AIF1", "APOE", "MRC1"],
    "DC": ["CLEC9A", "XCR1", "BATF3", "CD1C", "FCER1A", "CLEC10A", "LILRA4", "IRF7"],
    "Langerhans": ["CD207", "CD1A", "CD1E"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "KIT", "CMA1"],
    "Neutrophil": ["FCGR3B", "CSF3R", "CXCR2", "CEACAM3"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA", "FAP", "POSTN"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "FLT1", "RAMP2", "EGFL7"],
    "LymphaticEndo": ["LYVE1", "PROX1", "PDPN", "CCL21", "MMRN1"],
    "SmoothMuscle": ["MYH11", "ACTA2", "TAGLN", "DES", "CNN1", "MYLK"],
    "Pericyte": ["RGS5", "NOTCH3", "KCNJ8", "PDGFRB", "ABCC9"],
}
CONTROL = ("Intergenic_Region", "NegControl", "NegPrb", "BLANK", "Blank", "Deprecated",
           "Unassigned", "Codeword", "antisense", "genomic_control", "Genomic_Control")


# Control probes and codewords excluded from typing matrices and cell tables.
BUILD_CONTROL = ("NegControlProbe_", "antisense_", "NegControlCodeword", "BLANK_", "DeprecatedCodeword_",
                 "UnassignedCodeword_", "Intergenic_Region", "genomic_control", "Genomic_Control")


def is_control(gene: str) -> bool:
    return any(c in gene for c in CONTROL)


def shared_subsample(segger_cells, tenx_cells, n: int = 100_000, seed: int = 0) -> np.ndarray:
    """Sorted random subsample of the cells present in both segmentations."""
    shared = np.sort(np.intersect1d(np.asarray(segger_cells, str), np.asarray(tenx_cells, str)))
    take = np.random.default_rng(seed).choice(len(shared), size=min(n, len(shared)), replace=False)
    return np.sort(shared[take])


def embed(X, cells, genes, min_counts: int = 50, min_genes: int = 3, resolution: float = 1.0):
    """QC, log-normalisation, 2,000 HVGs, PCA, kNN graph, Leiden and UMAP.

    Returns the embedded AnnData (QC-passing cells) and the log-normalised matrix over
    all genes of those cells.
    """
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=X)
    a.obs_names = np.asarray(cells, str)
    a.var_names = np.asarray(genes, str)
    a.obs["n_counts"] = np.asarray(X.sum(1)).ravel()
    a.obs["n_genes"] = np.asarray((X > 0).sum(1)).ravel()
    a = a[(a.obs["n_counts"] >= min_counts) & (a.obs["n_genes"] >= min_genes)].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    lognorm = a.X.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50)
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=50)
    sc.tl.leiden(a, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(a, random_state=0)
    return a, lognorm


def lognorm(X):
    """log1p of counts per 10,000, in float64."""
    X = sparse.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1.0
    Y = sparse.csr_matrix(X.multiply(1e4 / lib[:, None]))
    Y.data = np.log1p(Y.data)
    return Y


def sctype_clusters(lognorm, genes, clusters, markers=MARKERS, score_min: float = 0.5, margin_min: float | None = 0.15):
    """ScType scores of cluster-mean profiles, z-scored across clusters.

    Marker weights are 1/k for a marker shared by k types. A cluster is ``Unknown`` when
    its best score is below ``score_min`` or, if ``margin_min`` is set, when the best
    score leads the second by less than ``margin_min``.

    Returns the per-cell label and a per-cluster table.
    """
    gi = {g: i for i, g in enumerate(genes)}
    types = [t for t in markers if any(g in gi for g in markers[t])]
    ngene: dict[str, int] = {}
    for t in markers:
        for g in markers[t]:
            ngene[g] = ngene.get(g, 0) + 1
    cols = sorted({gi[g] for t in types for g in markers[t] if g in gi})
    cpos = {c: i for i, c in enumerate(cols)}
    M = lognorm[:, cols]
    M = np.asarray(M.todense()) if sparse.issparse(M) else np.asarray(M)
    clusters = np.asarray(clusters)
    cl = np.unique(clusters)
    prof = np.vstack([M[clusters == c].mean(0) for c in cl])
    Z = (prof - prof.mean(0)) / (prof.std(0) + 1e-9)
    scores = np.full((len(cl), len(types)), -1e9)
    for j, t in enumerate(types):
        idx = [cpos[gi[g]] for g in markers[t] if g in gi]
        w = np.array([1.0 / ngene[g] for g in markers[t] if g in gi])
        scores[:, j] = (Z[:, idx] * w).sum(1) / w.sum()
    srt = np.sort(scores, 1)
    top, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    lab = np.array(types, dtype=object)[scores.argmax(1)]
    confident = top >= score_min
    if margin_min is not None:
        confident &= margin >= margin_min
    lab = np.where(confident, lab, "Unknown")
    table = pd.DataFrame({"cluster": cl.astype(str), "n": [int((clusters == c).sum()) for c in cl], "cell_type": lab,
                          "score": top, "margin": margin, "confident": confident})
    return pd.Series(clusters.astype(str)).map(dict(zip(cl.astype(str), lab))).to_numpy(), table


def exclusive_markers(markers=MARKERS) -> pd.Series:
    """Gene -> cell type for the curated markers that mark exactly one type."""
    owners: dict[str, list[str]] = {}
    for t, genes in markers.items():
        for g in genes:
            owners.setdefault(g, []).append(t)
    return pd.Series({g: ts[0] for g, ts in owners.items() if len(ts) == 1}, name="cell_type")


def call_lineage(counts: pd.DataFrame, composition: pd.Series, min_evidence: float = 3.0, margin: float = 2.0,
                 pseudo: float = 0.5) -> pd.Series:
    """Cell type with the highest marker enrichment over the expected composition.

    ``counts`` has columns cell, lin, n (marker transcripts per cell and type). The
    enrichment of type k in a cell is (n_k + pseudo) / (n * p_k + pseudo) for ``composition``
    p. A call needs at least ``min_evidence`` transcripts of the winning type and a
    ``margin``-fold lead in enrichment; otherwise the cell is ``uncalled``.
    """
    w = counts.pivot_table(index="cell", columns="lin", values="n", aggfunc="sum", fill_value=0.0)
    w = w.reindex(columns=sorted(composition.index), fill_value=0.0)
    n = w.to_numpy()
    exp = np.outer(n.sum(1), composition.reindex(w.columns).to_numpy())
    e = (n + pseudo) / (exp + pseudo)
    e[n == 0] = 0.0
    order = np.argsort(-e, axis=1)
    r = np.arange(len(w))
    top, second = order[:, 0], order[:, 1]
    ok = (n[r, top] >= min_evidence) & (e[r, top] >= margin * np.maximum(e[r, second], 1e-12))
    return pd.Series(np.where(ok, np.asarray(w.columns)[top], "uncalled"), index=w.index, name="cell_type")


def marker_pcs(X, n_pcs: int = 20, seed: int = 0) -> np.ndarray:
    """PCA of log1p(CP10K) marker counts after scaling each marker to unit variance."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    total = np.asarray(X.sum(1)).ravel()
    total[total == 0] = 1.0
    y = StandardScaler().fit_transform(np.log1p(sparse.csr_matrix(X).multiply(1e4 / total[:, None]).toarray()))
    return PCA(n_components=n_pcs, random_state=seed).fit_transform(y)


def separation(pcs: np.ndarray, labels, n_sub: int = 12_000, seeds: int = 3, n_neighbors: int = 15, n_boot: int = 200):
    """Silhouette and macro kNN label purity of ``labels`` in ``pcs``.

    Silhouette is averaged over ``seeds`` random subsets of ``n_sub`` cells. Purity is the
    share of each cell's ``n_neighbors`` nearest neighbours with its label, averaged per
    label and then over labels; ``n_boot`` bootstrap resamples of cells give a 95% interval.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    sil = [silhouette_score(pcs[r], labels[r]) for r in
           [np.random.default_rng(s).choice(len(labels), min(n_sub, len(labels)), False) for s in range(seeds)]]
    codes = pd.factorize(labels)[0]
    _, ind = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(pcs).kneighbors(pcs)
    purity = (codes[ind[:, 1:]] == codes[:, None]).mean(1)
    per_label = pd.DataFrame({"l": labels, "p": purity}).groupby("l")["p"].mean()
    rng = np.random.default_rng(0)
    boot = [pd.DataFrame({"l": labels[s], "p": purity[s]}).groupby("l")["p"].mean().mean()
            for s in (rng.integers(0, len(labels), len(labels)) for _ in range(n_boot))]
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"silhouette": float(np.mean(sil)), "knn_purity": float(per_label.mean()), "knn_purity_low": float(low),
            "knn_purity_high": float(high), "n": len(labels), "k": int(pd.Series(labels).nunique())}, per_label


def type_profiles(X, labels) -> tuple[np.ndarray, list[str]]:
    """Mean raw counts per label (labels sorted), as float64."""
    labels = np.asarray(labels)
    types = sorted(pd.unique(labels))
    return np.vstack([np.asarray(X[labels == t].mean(0)).ravel() for t in types]).astype(np.float64), types


def cosine_types(X, profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-matching profile by cosine similarity, and the typing confidence.

    Confidence is 1 - H(p) / log K with p proportional to the squared positive cosine
    similarities to the K profiles.
    """
    ref = np.asarray(profiles, np.float64)
    ref_norm = np.linalg.norm(ref, axis=1)
    ref_norm[~np.isfinite(ref_norm) | (ref_norm <= 0)] = 1.0
    X = sparse.csr_matrix(X)
    cell_norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    cell_norm[~np.isfinite(cell_norm) | (cell_norm <= 0)] = 1.0
    sim = np.asarray(X @ ref.T, np.float64) / cell_norm[:, None] / ref_norm[None, :]
    mass = np.clip(sim, 0.0, None) ** 2
    den = mass.sum(1)
    p = np.full_like(mass, 1.0 / sim.shape[1])
    ok = den > 0
    p[ok] = mass[ok] / den[ok, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), 1)
    return sim.argmax(1), 1.0 - entropy / math.log(sim.shape[1])


def rank_reference_genes(X, labels, genes) -> dict[str, pd.DataFrame]:
    """One-vs-rest Wilcoxon test per label on log1p(CP10K) counts, with detection rates.

    Returns per label a table with names, scores (z), logfoldchanges, pvals_adj and
    prev, the fraction of that label's cells with at least one count.
    """
    import anndata as ad
    import scanpy as sc

    labels = np.asarray(labels, str)
    a = ad.AnnData(X=sparse.csr_matrix(X, dtype=np.float32))
    a.var_names = np.asarray(genes, str)
    a.obs["label"] = pd.Categorical(labels)
    raw = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    types = sorted(set(labels))
    sc.tl.rank_genes_groups(a, groupby="label", groups=types, method="wilcoxon", use_raw=False)
    out = {}
    for t in types:
        df = sc.get.rank_genes_groups_df(a, group=t)
        prev = pd.Series(np.asarray((raw[labels == t] > 0).mean(0)).ravel(), index=a.var_names)
        out[t] = df.assign(prev=df["names"].map(prev).to_numpy())
    return out


def reference_markers(ranked: dict[str, pd.DataFrame], positive: bool = True, n: int = 12, padj_max: float = 0.01,
                      logfc_min: float = 1.0, prev_min: float = 0.5, prev_max: float = 0.1) -> dict[str, list[str]]:
    """Top-``n`` positive (enriched, detected in >= ``prev_min`` of the label's cells) or
    negative (depleted, detected in < ``prev_max``) markers per label, ranked by z-score."""
    out = {}
    for t, df in ranked.items():
        if positive:
            sig = df[(df.pvals_adj < padj_max) & (df.logfoldchanges > logfc_min) & (df.prev >= prev_min)]
            sig = sig.sort_values("scores", ascending=False)
        else:
            sig = df[(df.pvals_adj < padj_max) & (df.logfoldchanges < -logfc_min) & (df.prev < prev_max)]
            sig = sig.sort_values("scores", ascending=True)
        out[t] = [str(g) for g in sig["names"].head(n)]
    return out


def marker_recall(X, genes, host, types, markers: dict[str, list[str]], profiles: np.ndarray, available=None):
    """Positive marker recall per cell (%).

    For a cell of type t, the share of the reference weight (mean count in t) of t's
    markers that the cell holds at least one transcript of, among the markers that are
    ``available`` near the cell (all markers when ``available`` is None). ``available``
    is a boolean cells x genes matrix aligned to ``genes``. Cells without an available
    marker are NaN.
    """
    col = {g: i for i, g in enumerate(genes)}
    X = sparse.csr_matrix(X)
    captured = X > 0
    vals = np.full(X.shape[0], np.nan)
    for ti, t in enumerate(types):
        mk = np.array([col[g] for g in markers.get(t, []) if g in col], dtype=np.int64)
        rows = np.where(host == ti)[0]
        if mk.size == 0 or rows.size == 0:
            continue
        w = profiles[ti, mk]
        cap = captured[rows][:, mk].toarray()
        av = np.ones_like(cap) if available is None else sparse.csr_matrix(available)[rows][:, mk].toarray().astype(bool)
        den = av @ w
        num = (av & cap) @ w
        ok = den > 0
        vals[rows[ok]] = 100.0 * num[ok] / den[ok]
    return vals


def weighted_quantile(values, weights, q: float) -> float:
    """Quantile of ``values`` under ``weights``, interpolated at the mid-point of each weight."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, v))


def iter_assigned(transcripts, segger, columns=("cell_id", "feature_name", "x_location", "y_location"),
                  chunk_size: int = 25_000_000):
    """Transcripts scored by Segger with both assignments, in consecutive row chunks.

    The Segger table is joined to ``transcripts.parquet`` on ``row_index``, the row
    position in the transcript file, one block of ``chunk_size`` rows at a time so that a
    section of 10^9 transcripts never has to be held in memory. Each chunk carries
    row_index, the requested transcript ``columns`` and segger_cell_id,
    segger_similarity and similarity_threshold.
    """
    n = pq.ParquetFile(transcripts).metadata.num_rows
    seg = pl.scan_parquet(segger).select(["row_index", "segger_cell_id", "segger_similarity", "similarity_threshold"])
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        s = seg.filter(pl.col("row_index").is_between(start, stop, closed="left")).collect()
        if s.height == 0:
            continue
        t = (pl.scan_parquet(transcripts).slice(start, stop - start).select(list(columns))
             .with_row_index("row_index", offset=start).collect())
        yield t.with_columns(pl.col("row_index").cast(pl.Int64)).join(s, on="row_index", how="inner")


def marker_points(transcripts, genes, min_qv: float = 20.0, chunk_size: int = 50_000_000):
    """Positions of every transcript of ``genes`` with quality value >= ``min_qv``, assigned or not.

    Returns gene codes (position in ``genes``) and an (n, 2) float32 array of x, y.
    """
    genes = list(genes)
    enum = pl.Enum(genes)
    n_rows = pq.ParquetFile(transcripts).metadata.num_rows
    codes, xy = [], []
    for start in range(0, n_rows, chunk_size):
        t = (pl.scan_parquet(transcripts).slice(start, chunk_size)
             .filter(pl.col("feature_name").is_in(genes) & (pl.col("qv") >= min_qv))
             .select(pl.col("feature_name").cast(enum).to_physical().cast(pl.UInt16).alias("g"), "x_location", "y_location")
             .collect())
        codes.append(t["g"].to_numpy())
        xy.append(t.select(["x_location", "y_location"]).to_numpy().astype(np.float32))
    return np.concatenate(codes), np.concatenate(xy)


def marker_availability(codes, xy, centroids: np.ndarray, n_genes: int, radius: float = 10.0):
    """Genes with at least one transcript within ``radius`` µm of each centroid.

    ``codes`` and ``xy`` come from :func:`marker_points`. Returns a boolean cells x genes
    matrix whose columns follow the gene list given there.
    """
    from scipy.spatial import cKDTree

    q = np.asarray(centroids, np.float64)
    q = np.where(np.isfinite(q).all(1)[:, None], q, 1e12)
    rows, cols = [], []
    for g in range(n_genes):
        pts = xy[codes == g].astype(np.float64)
        if len(pts) == 0:
            continue
        hit = np.nonzero(cKDTree(pts).query_ball_point(q, r=radius, return_length=True) > 0)[0]
        rows.append(hit)
        cols.append(np.full(len(hit), g))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(q), n_genes))


def cell_code(column: str = "cell_id") -> pl.Expr:
    """Xenium cell id ('aaaabmoe-1') as the unsigned integer its eight letters encode."""
    return (pl.col(column).str.slice(0, 8).str.replace_many(list("abcdefghijklmnop"), list("0123456789abcdef"))
            .str.to_integer(base=16).cast(pl.UInt32))


def raw_pass(transcripts, genes, workdir, markers=MARKERS, min_qv: float = 20.0, chunk_size: int = 10_000_000,
             n_buckets: int = 8):
    """Everything taken from ``transcripts.parquet`` alone, in one pass.

    * the nuclear reference: transcripts of genes (``is_gene``) with quality value >=
      ``min_qv`` in a 10x cell, split by ``overlaps_nucleus``; per-gene nuclear and
      cytoplasmic counts, and unique (nucleus, gene) keys written to ``workdir`` in
      ``n_buckets`` shards;
    * intra-nuclear transcripts of the exclusive markers per 10x cell and gene
      (cell, gene, n), for the nuclear anchor;
    * 10x Cell transcripts with quality value >= 30 outside control probes per cell,
      with coordinate sums (cell_id, n, sx, sy).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    genes = list(genes)
    gene_table = pl.DataFrame({"feature_name": genes, "gene": np.arange(len(genes), dtype=np.uint16)})
    lin_genes = list(exclusive_markers(markers).index)
    ctrl = "|".join(BUILD_CONTROL)
    n_rows = pq.ParquetFile(transcripts).metadata.num_rows
    n_nuc = np.zeros(len(genes), np.int64)
    n_cyto = np.zeros(len(genes), np.int64)
    anchors, q30 = [], []
    for k, start in enumerate(range(0, n_rows, chunk_size)):
        t = (pl.scan_parquet(transcripts).slice(start, chunk_size)
             .select(["cell_id", "feature_name", "x_location", "y_location", "overlaps_nucleus", "qv", "is_gene"])
             .filter(pl.col("cell_id") != "UNASSIGNED").collect())
        q30.append(t.filter((pl.col("qv") >= 30) & ~pl.col("feature_name").str.contains(ctrl))
                   .group_by("cell_id").agg(pl.len().alias("n"), pl.col("x_location").cast(pl.Float64).sum().alias("sx"),
                                            pl.col("y_location").cast(pl.Float64).sum().alias("sy")))
        t = t.filter(pl.col("is_gene") & (pl.col("qv") >= min_qv))
        anchors.append(t.filter((pl.col("overlaps_nucleus") == 1) & pl.col("feature_name").is_in(lin_genes))
                       .group_by(["cell_id", "feature_name"]).agg(pl.len().alias("n")))
        t = (t.join(gene_table, on="feature_name", how="inner")
             .select(cell_code().alias("cell"), "gene", (pl.col("overlaps_nucleus") == 1).alias("nuc")))
        gene = t["gene"].to_numpy()
        nuc = t["nuc"].to_numpy()
        n_nuc += np.bincount(gene[nuc], minlength=len(genes))
        n_cyto += np.bincount(gene[~nuc], minlength=len(genes))
        keys = np.unique((t["cell"].to_numpy()[nuc].astype(np.uint64) << np.uint64(16)) | gene[nuc].astype(np.uint64))
        shard = (keys >> np.uint64(16)) % np.uint64(n_buckets)
        for b in range(n_buckets):
            np.save(workdir / f"keys_{b}_{k:04d}.npy", keys[shard == b])
    counts = pd.DataFrame({"gene": genes, "n_nuc": n_nuc, "n_cyto": n_cyto})
    anchor = _sum_by(anchors, ["cell_id", "feature_name"]).rename({"cell_id": "cell", "feature_name": "gene"}).to_pandas()
    return counts, anchor, _sum_by(q30, "cell_id")


def _shard_keys(workdir, b: int) -> np.ndarray:
    parts = sorted(Path(workdir).glob(f"keys_{b}_*.npy"))
    return np.unique(np.concatenate([np.load(p) for p in parts]))


def exclusive_nuclear_pairs(workdir, counts: pd.DataFrame, genes=None, n_buckets: int = 8, min_gene_count: int = 10,
                            min_occurrence: float = 0.01, max_jaccard: float = 0.10, max_pairs: int = 500,
                            block: int = 8192) -> tuple[pd.DataFrame, dict, list]:
    """Semi-exclusive gene pairs in the nuclear reference, as in ``segger validate``.

    Genes need ``min_gene_count`` nuclear transcripts; pairs need both genes in at least
    ``min_occurrence`` of the nuclei and a nuclear co-occurrence Jaccard below
    ``max_jaccard``. The ``max_pairs`` most exclusive pairs are kept. Each pair is
    weighted by sqrt(occ_a occ_b) / sqrt((1 + r_a)(1 + r_b)), with r the gene's
    cytoplasmic to nuclear transcript ratio. ``genes`` restricts the nuclear gene list.

    Returns the pairs, a summary and the nuclear gene list.
    """
    whitelist = counts["n_nuc"].to_numpy() >= min_gene_count
    if genes is not None:
        whitelist &= counts["gene"].isin(set(genes)).to_numpy()
    occ = np.zeros(len(counts), np.float64)
    nuclei = 0
    for b in range(n_buckets):
        keys = _shard_keys(workdir, b)
        g = (keys & np.uint64(0xFFFF)).astype(np.int64)
        keep = whitelist[g]
        occ += np.bincount(g[keep], minlength=len(counts))
        nuclei += len(np.unique(keys[keep] >> np.uint64(16)))
    min_occ = max(3, int(nuclei * float(min_occurrence)))
    ok = np.nonzero(whitelist & (occ >= min_occ))[0]
    pos = np.full(len(counts), -1, np.int64)
    pos[ok] = np.arange(len(ok))
    co = np.zeros((len(ok), len(ok)), np.float64)
    for b in range(n_buckets):
        keys = _shard_keys(workdir, b)
        g = pos[(keys & np.uint64(0xFFFF)).astype(np.int64)]
        keys, g = keys[g >= 0], g[g >= 0]
        _, rows = np.unique(keys >> np.uint64(16), return_inverse=True)
        B = sparse.csr_matrix((np.ones(len(g), np.float32), (rows, g)), shape=(rows.max() + 1, len(ok)))
        for s in range(0, B.shape[0], block):
            d = B[s:s + block].toarray()
            co += d.T @ d
    o = occ[ok]
    jac = co / (o[:, None] + o[None, :] - co + 1e-12)
    np.fill_diagonal(jac, 1.0)
    i, j = np.where(np.triu(jac < max_jaccard, 1))
    score = jac[i, j]
    order = np.argsort(score)[:max_pairs]
    i, j = i[order], j[order]
    gi, gj = ok[i], ok[j]
    ratio = counts["n_cyto"].to_numpy() / np.maximum(counts["n_nuc"].to_numpy(), 1.0)
    genes = counts["gene"].to_numpy()
    pairs = pd.DataFrame({"g1": genes[gi], "g2": genes[gj], "nuclear_jaccard": jac[i, j],
                          "pair_weight": np.sqrt(occ[gi] * occ[gj]) / np.sqrt((1 + ratio[gi]) * (1 + ratio[gj]))})
    info = {"nuclei": nuclei, "whitelist": int(whitelist.sum()), "gene_ok": len(ok), "min_occurrence": min_occ,
            "candidate_pairs": int(len(score)), "nuclear_transcripts": int(counts["n_nuc"].to_numpy()[whitelist].sum())}
    return pairs, info, list(counts["gene"].to_numpy()[whitelist])


def pair_coexpression(X, genes, pairs: pd.DataFrame) -> np.ndarray:
    """Soft Jaccard of each pair over cells: sum(min(a, b)) / sum(max(a, b)).

    ``X`` holds per-cell counts divided by each cell's total over the nuclear gene list.
    """
    col = {g: i for i, g in enumerate(genes)}
    X = sparse.csc_matrix(X)
    a = X[:, [col[g] for g in pairs["g1"]]]
    b = X[:, [col[g] for g in pairs["g2"]]]
    return a.minimum(b).sum(0).A1 / (a.maximum(b).sum(0).A1 + 1e-12)


def _with_thresholds(t: pl.DataFrame, thresholds: dict[str, float] | None) -> pl.DataFrame:
    """Add ``thr``, the per-gene threshold of ``thresholds`` or Segger's own ``similarity_threshold``."""
    if thresholds is None:
        return t.with_columns(pl.col("similarity_threshold").alias("thr"))
    table = pl.DataFrame({"feature_name": list(thresholds), "thr": list(thresholds.values())})
    return t.join(table, on="feature_name", how="left")


def _sum_by(frames: list[pl.DataFrame], key) -> pl.DataFrame:
    key = [key] if isinstance(key, str) else list(key)
    df = pl.concat(frames)
    return df.group_by(key).agg(pl.exclude(key).sum()).sort(key)


def cell_summaries(transcripts, segger, thresholds: dict[str, float] | None = None, block: float = 500.0,
                   chunk_size: int = 20_000_000):
    """Per-cell transcript counts and centroids of both segmentations, and per-block coverage.

    In one pass over the Segger table joined to the transcripts:

    * ``kept``: Segger transcripts with similarity at or above their own threshold, per cell;
    * ``kept_table``: the same with the per-gene ``thresholds`` given as a table (Segger's own
      threshold when ``thresholds`` is None);
    * ``tenx``: 10x Cell transcripts in Segger's universe, per cell;
    * ``argmax_q30``: transcripts with quality value >= 30 outside control probes per
      Segger cell before thresholding, with coordinate sums;
    * ``blocks``: transcripts, Segger (tabulated thresholds) and 10x Cell assignments per
      ``block`` µm tile;
    * ``genes``: transcripts per gene in Segger's universe.
    """
    ctrl = "|".join(BUILD_CONTROL)
    out = {k: [] for k in ("kept", "kept_table", "tenx", "argmax_q30", "blocks", "genes")}
    cols = ("cell_id", "feature_name", "x_location", "y_location", "qv")
    for t in iter_assigned(transcripts, segger, columns=cols, chunk_size=chunk_size):
        t = _with_thresholds(t, thresholds)
        seg = pl.col("segger_cell_id").is_not_null()
        own = seg & (pl.col("segger_similarity") >= pl.col("similarity_threshold"))
        table = seg & (pl.col("segger_similarity") >= pl.col("thr"))
        out["kept"].append(t.filter(own).group_by("segger_cell_id").agg(pl.len().alias("n")))
        out["kept_table"].append(t.filter(table).group_by("segger_cell_id").agg(pl.len().alias("n")))
        out["tenx"].append(t.filter(pl.col("cell_id") != "UNASSIGNED").group_by("cell_id").agg(pl.len().alias("n")))
        out["argmax_q30"].append(
            t.filter(seg & (pl.col("qv") >= 30) & ~pl.col("feature_name").str.contains(ctrl))
            .group_by("segger_cell_id").agg(pl.len().alias("n"), pl.col("x_location").cast(pl.Float64).sum().alias("sx"),
                                            pl.col("y_location").cast(pl.Float64).sum().alias("sy")))
        out["blocks"].append(
            t.with_columns((pl.col("x_location") // block).cast(pl.Int32).alias("bx"),
                           (pl.col("y_location") // block).cast(pl.Int32).alias("by"))
            .group_by(["bx", "by"]).agg(pl.len().alias("n_tx"), table.sum().alias("n_segger"),
                                        (pl.col("cell_id") != "UNASSIGNED").sum().alias("n_tenx")))
        out["genes"].append(t.group_by("feature_name").agg(pl.len().alias("n")))
    keys = {"kept": "segger_cell_id", "kept_table": "segger_cell_id", "tenx": "cell_id", "argmax_q30": "segger_cell_id",
            "blocks": ["bx", "by"], "genes": "feature_name"}
    return {k: _sum_by(v, keys[k]) for k, v in out.items()}


def subset_and_pair_counts(transcripts, segger, subset, genes, thresholds: dict[str, float], pair_genes, whitelists: dict,
                           workdir, chunk_size: int = 20_000_000):
    """Counts needed on top of the whole-section cell tables, in one joined pass.

    * cell x gene triplets (cidx, gidx, n) of the ``subset`` cells over ``genes`` (gidx =
      position), for 10x Cell and for Segger with its own thresholds (``segger``) and with
      the tabulated per-gene ``thresholds`` (``segger_table``);
    * for every cell of each segmentation (as :func:`cell_code`), totals over each gene
      list in ``whitelists`` and counts of the ``pair_genes`` (gidx).

    Each block is written to ``workdir`` and the shards are summed at the end, so memory
    stays bounded by one block.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sub = pl.DataFrame({"cell": np.asarray(subset, str), "cidx": np.arange(len(subset), dtype=np.int32)})
    gi = pl.DataFrame({"feature_name": list(genes), "gidx": np.arange(len(genes), dtype=np.int32)})
    wl = pl.DataFrame({"feature_name": sorted(set().union(*whitelists.values()))}).with_columns(
        [pl.col("feature_name").is_in(list(v)).alias(f"wl_{k}") for k, v in whitelists.items()])
    pg = set(pair_genes)
    pair_idx = [i for i, g in enumerate(genes) if g in pg]
    arms = ("tenx", "segger", "segger_table")
    for k, t in enumerate(iter_assigned(transcripts, segger, columns=("cell_id", "feature_name"), chunk_size=chunk_size)):
        t = _with_thresholds(t.join(gi, on="feature_name", how="left"), thresholds).join(wl, on="feature_name", how="left")
        seg = pl.col("segger_cell_id").is_not_null()
        sel = {"tenx": (pl.col("cell_id") != "UNASSIGNED", "cell_id"),
               "segger": (seg & (pl.col("segger_similarity") >= pl.col("similarity_threshold")), "segger_cell_id"),
               "segger_table": (seg & (pl.col("segger_similarity") >= pl.col("thr")), "segger_cell_id")}
        for arm, (keep, col) in sel.items():
            a = t.filter(keep).select(col, "gidx", *[f"wl_{w}" for w in whitelists])
            (a.filter(pl.col("gidx").is_not_null()).join(sub, left_on=col, right_on="cell", how="inner")
             .group_by(["cidx", "gidx"]).agg(pl.len().cast(pl.UInt32).alias("n"))
             .write_parquet(workdir / f"tri_{arm}__{k:04d}.parquet"))
            a = a.with_columns(cell_code(col).alias("cell"))
            (a.group_by("cell").agg([pl.col(f"wl_{w}").fill_null(False).sum().cast(pl.UInt32).alias(w) for w in whitelists])
             .write_parquet(workdir / f"tot_{arm}__{k:04d}.parquet"))
            (a.filter(pl.col("gidx").is_in(pair_idx)).group_by(["cell", "gidx"]).agg(pl.len().cast(pl.UInt32).alias("n"))
             .write_parquet(workdir / f"pair_{arm}__{k:04d}.parquet"))
    out = {}
    for kind, key in (("tri", ["cidx", "gidx"]), ("tot", ["cell"]), ("pair", ["cell", "gidx"])):
        for arm in arms:
            out[f"{kind}_{arm}"] = (pl.scan_parquet(workdir / f"{kind}_{arm}__*.parquet").group_by(key)
                                    .agg(pl.exclude(key).sum()).sort(key).collect(engine="streaming"))
    return out


def subset_matrix(tri: pl.DataFrame, n_cells: int, universe, axis, compact: bool = False):
    """Cell x gene counts of the subset cells from (cidx, gidx, n) triplets.

    ``universe`` is the gene list the triplets index; ``axis`` the genes counted in each
    cell's total. The matrix keeps the non-control genes of ``axis`` (``compact``: only
    those with a count).

    Returns the matrix, its genes and the per-cell totals over ``axis``.
    """
    genes = np.asarray(universe)
    in_axis = np.isin(genes, list(axis))
    control = np.array([is_control(g) for g in genes])
    c, g, n = tri["cidx"].to_numpy(), tri["gidx"].to_numpy(), tri["n"].to_numpy().astype(np.float64)
    k = in_axis[g]
    totals = np.bincount(c[k], weights=n[k], minlength=n_cells)
    keep = in_axis & ~control
    if compact:
        keep &= np.bincount(g[k], minlength=len(genes)) > 0
    col = np.full(len(genes), -1)
    col[keep] = np.arange(keep.sum())
    k = col[g] >= 0
    X = sparse.coo_matrix((n[k].astype(np.float32), (c[k], col[g[k]])), shape=(n_cells, int(keep.sum()))).tocsr()
    return X, genes[keep], totals


def crop(transcripts, segger, boxes, thresholds: dict[str, float] | None = None, chunk_size: int = 50_000_000) -> pl.DataFrame:
    """Transcripts inside ``boxes`` ((xmin, ymin, xmax, ymax) in µm) with their Segger assignment.

    Returns row_index, x, y, feature_name, segger_cell_id, segger_similarity and ``keep``:
    the transcript has a Segger cell and a similarity at or above its gene's threshold
    (``thresholds``, or Segger's own when None).
    """
    inside = pl.any_horizontal([pl.col("x_location").is_between(b[0], b[2], closed="left")
                                & pl.col("y_location").is_between(b[1], b[3], closed="left") for b in boxes])
    n_rows = pq.ParquetFile(transcripts).metadata.num_rows
    parts = []
    for start in range(0, n_rows, chunk_size):
        t = (pl.scan_parquet(transcripts).slice(start, chunk_size).with_row_index("row_index", offset=start)
             .filter(inside).select(pl.col("row_index").cast(pl.Int64), "x_location", "y_location", "feature_name").collect())
        if t.height:
            parts.append(t)
    tx = pl.concat(parts)
    seg = (pl.scan_parquet(segger).filter(pl.col("row_index").is_in(tx["row_index"].implode()))
           .select("row_index", "segger_cell_id", "segger_similarity", "similarity_threshold").collect())
    tx = _with_thresholds(tx.join(seg, on="row_index", how="left"), thresholds)
    keep = pl.col("segger_cell_id").is_not_null() & (pl.col("segger_similarity") >= pl.col("thr"))
    return tx.with_columns(keep.fill_null(False).alias("keep")).drop("thr", "similarity_threshold")
