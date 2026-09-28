"""Datasets of the paper and their download.

Raw iST data come from the vendors (10x Genomics) or the lab's public bucket.
The segmentation results of all methods and the single-cell references used for
cell typing are hosted in the same bucket. :func:`fetch` downloads what a
notebook needs into ``data_dir/<dataset>/`` and skips files that are already
there.

From 10x Genomics ZIP bundles, only the members used by segger and the notebooks
are read, using HTTP range requests, so the full archive (13 to 41 GB) is never
downloaded.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import posixpath
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BUCKET = "https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger"
TENX = "https://cf.10xgenomics.com/samples/xenium"

# Members of a Xenium output bundle needed by `segger segment` and the notebooks.
XENIUM_MEMBERS = (
    "experiment.xenium",
    "gene_panel.json",
    "metrics_summary.csv",
    "transcripts.parquet",
    "cells.parquet",
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
    "cell_feature_matrix.h5",
)
MORPHOLOGY_FOCUS = tuple(f"morphology_focus/morphology_focus_000{i}.ome.tif" for i in range(4))
# Atera bundles have no gene_panel.json; the notebooks do not read the cell x gene matrix.
ATERA_MEMBERS = (
    "experiment.xenium",
    "metrics_summary.csv",
    "transcripts.parquet",
    "cells.parquet",
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
)


@dataclass(frozen=True)
class Remote:
    """A remote file, or a set of members of a remote archive.

    Parameters
    ----------
    url
        HTTPS location.
    target
        Local path relative to the dataset directory: a file, or the directory
        the archive members are extracted to.
    members
        Members to extract from a ZIP archive, or files to download from a
        directory (``url`` ending in ``/``); None for a single file.
    """

    url: str
    target: str
    members: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Dataset:
    """A dataset of the paper: a title, where the raw data come from, and its files."""

    title: str
    source: str
    files: dict[str, Remote] = field(default_factory=dict)


DATASETS: dict[str, Dataset] = {
    "xenium_crc": Dataset(
        title="Human colorectal cancer, Xenium with multimodal cell segmentation staining",
        source=(
            "10x Genomics, Human Colon Preview Data (Xenium Human Colon Gene Expression Panel), "
            "section Human_Colon_Cancer_P1_CRC_Add_on (Oliveira et al., Nat. Genet. 2025)"
        ),
        files={
            "xenium": Remote(
                url=f"{TENX}/2.0.0/Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE/"
                "Xenium_V1_Human_Colon_Cancer_P1_CRC_Add_on_FFPE_outs.zip",
                target="xenium",
                members=XENIUM_MEMBERS + MORPHOLOGY_FOCUS,
            ),
            "segger": Remote(f"{BUCKET}/paper/xenium_crc/segger_segmentation.parquet", "segger/segger_segmentation.parquet"),
            "segmentations": Remote(f"{BUCKET}/paper/xenium_crc/segmentations.parquet", "segmentations.parquet"),
            "reference": Remote(f"{BUCKET}/paper/xenium_crc/human_crc_flex_reference.h5ad", "human_crc_flex_reference.h5ad"),
            "benchmark": Remote(f"{BUCKET}/paper/benchmark/benchmark_scores.tsv", "benchmark_scores.tsv"),
        },
    ),
    "xenium_nsclc": Dataset(
        title="Human non-small-cell lung cancer, Xenium with NaK-ATPase membrane staining",
        source="Generated for this study; full archive at segger/xenium_nsclc.tar.gz (66 GB)",
        files={
            "xenium": Remote(f"{BUCKET}/paper/xenium_nsclc/xenium/", "xenium", members=XENIUM_MEMBERS),
            "cellpose": Remote(
                f"{BUCKET}/paper/xenium_nsclc/cellpose/",
                "cellpose",
                members=("cellpose_mask_polygons.parquet", "segmentation_image.npy"),
            ),
            "celltypist": Remote(
                f"{BUCKET}/paper/xenium_nsclc/celltypist/",
                "celltypist",
                members=("nsclc_celltypist_model.pkl", "nsclc_celltypist_finetype_model.pkl"),
            ),
            "atlas": Remote(f"{BUCKET}/paper/xenium_nsclc/core_nsclc_atlas_panel_only.h5ad", "core_nsclc_atlas_panel_only.h5ad"),
            "segger": Remote(f"{BUCKET}/paper/xenium_nsclc/segger_segmentation.parquet", "segger/segger_segmentation.parquet"),
            "segmentations": Remote(f"{BUCKET}/paper/xenium_nsclc/segmentations.parquet", "segmentations.parquet"),
            "area_sample": Remote(f"{BUCKET}/paper/xenium_nsclc/figure_4_area_sample.parquet", "figure_4_area_sample.parquet"),
            "cell_types": Remote(f"{BUCKET}/paper/xenium_nsclc/figure_4_cell_types.parquet", "figure_4_cell_types.parquet"),
            "benchmark": Remote(f"{BUCKET}/paper/benchmark/benchmark_scores.tsv", "benchmark_scores.tsv"),
        },
    ),
    "xenium_breast": Dataset(
        title="Human breast cancer, Xenium Prime 5K with 100 custom genes",
        source="10x Genomics, FFPE Human Breast Cancer with 5K Human Pan Tissue and Pathways Panel",
        files={
            "xenium": Remote(
                url=f"{TENX}/3.0.0/Xenium_Prime_Breast_Cancer_FFPE/Xenium_Prime_Breast_Cancer_FFPE_outs.zip",
                target="xenium",
                members=XENIUM_MEMBERS,
            ),
            "segger": Remote(f"{BUCKET}/paper/xenium_breast/segger_segmentation.parquet", "segger/segger_segmentation.parquet"),
            "segmentations": Remote(f"{BUCKET}/paper/xenium_breast/segmentations.parquet", "segmentations.parquet"),
            "reference": Remote(f"{BUCKET}/paper/xenium_breast/census_breast_reference.h5ad", "census_breast_reference.h5ad"),
        },
    ),
    "atera_cervical": Dataset(
        title="Human cervical squamous-cell carcinoma, Atera whole transcriptome (18,028 genes)",
        source="10x Genomics, Atera WTA Preview FFPE Human Cervical Cancer",
        files={
            "xenium": Remote(
                url="https://s3-us-west-2.amazonaws.com/10x.files/samples/atera/dev/WTA_Preview_FFPE_Cervical_Cancer/"
                "WTA_Preview_FFPE_Cervical_Cancer_outs.zip",
                target="xenium",
                members=ATERA_MEMBERS,
            ),
            "segger": Remote(f"{BUCKET}/paper/atera_cervical/segger_segmentation.parquet", "segger/segger_segmentation.parquet"),
            "split_plan": Remote(f"{BUCKET}/paper/atera_cervical/gene_split_plan.parquet", "segger/gene_split_plan.parquet"),
            "thresholds": Remote(f"{BUCKET}/paper/atera_cervical/segger_gene_thresholds.tsv", "segger_gene_thresholds.tsv"),
            "cell_types": Remote(f"{BUCKET}/paper/atera_cervical/", ".", members=("cell_types_v2.parquet", "cell_types_v3.parquet")),
        },
    ),
}


# md5 of every file as used in the paper, keyed by its path under ``data_dir``.
MD5 = {
    "xenium_crc/xenium/experiment.xenium": "31cfadbb7c6c78ac7757ed5f1405e1cb",
    "xenium_crc/xenium/gene_panel.json": "2d614474a7ca8e21c30734c7f4a1bfdf",
    "xenium_crc/xenium/metrics_summary.csv": "ed29d2ee1c3a86da9c2cb018cf6bbbec",
    "xenium_crc/xenium/transcripts.parquet": "b761a83214cba69f9a43c3d3a16ff4f4",
    "xenium_crc/xenium/cells.parquet": "78c37dbd6f9a89ddffd762371220134c",
    "xenium_crc/xenium/cell_boundaries.parquet": "3ef514e883735036d15b4f59fa3d99f3",
    "xenium_crc/xenium/nucleus_boundaries.parquet": "fe2d796151bc8b7ebc1cece7ebdbbb8f",
    "xenium_crc/xenium/cell_feature_matrix.h5": "08a78f8821a4c6b6ca18392ec16725ee",
    "xenium_crc/xenium/morphology_focus/morphology_focus_0000.ome.tif": "e241a6c187af20b4abd420c428231170",
    "xenium_crc/xenium/morphology_focus/morphology_focus_0001.ome.tif": "9e70fbd2456b04657caae721c34fd199",
    "xenium_crc/xenium/morphology_focus/morphology_focus_0002.ome.tif": "c914a5cf3bfe6da431c8c6722f0f4816",
    "xenium_crc/xenium/morphology_focus/morphology_focus_0003.ome.tif": "ed1f01df12274e37907a13968856699d",
    "xenium_crc/segger/segger_segmentation.parquet": "229b30a17facac7e36059a7ee6b0c41b",
    "xenium_crc/segmentations.parquet": "b496d9ad3d7d991fe0ea40a954b3680d",
    "xenium_crc/human_crc_flex_reference.h5ad": "36059e31f025e52ab0beab96664ce7e5",
    "xenium_crc/benchmark_scores.tsv": "8938a747f88821cd628f929928ac969e",
    "xenium_nsclc/xenium/experiment.xenium": "2cc4e33161d4a900fbc274d6c8b7081d",
    "xenium_nsclc/xenium/gene_panel.json": "b4a8ab5086a395079ab93981da1591f5",
    "xenium_nsclc/xenium/metrics_summary.csv": "60d9a6e457e05d182b59b99601aec5a9",
    "xenium_nsclc/xenium/transcripts.parquet": "1f61db1dda46b460b65628c5bfb43dd3",
    "xenium_nsclc/xenium/cells.parquet": "d565c8e364160419af14f44267f5e8b1",
    "xenium_nsclc/xenium/cell_boundaries.parquet": "1cdb0b9135cbbf43ad1df3f5e8d11a15",
    "xenium_nsclc/xenium/nucleus_boundaries.parquet": "204b098e4121162f8ddf6ffe99ae3c80",
    "xenium_nsclc/xenium/cell_feature_matrix.h5": "c5c763eb640e00d104b45a04eb9611c4",
    "xenium_nsclc/cellpose/cellpose_mask_polygons.parquet": "fc2579223b48c900cde94e67d4f49103",
    "xenium_nsclc/cellpose/segmentation_image.npy": "cd72833fe1640c6850a8107b9f469c04",
    "xenium_nsclc/celltypist/nsclc_celltypist_model.pkl": "8278e162a7582d463e8b164b1b7d4dc7",
    "xenium_nsclc/celltypist/nsclc_celltypist_finetype_model.pkl": "cb7553bd27422039e5d60a6a45796e9b",
    "xenium_nsclc/core_nsclc_atlas_panel_only.h5ad": "87f4c34033e66ca924ff45dd839f24fa",
    "xenium_nsclc/segger/segger_segmentation.parquet": "d1bc1131cd4ba5e8349e62b19d276741",
    "xenium_nsclc/segmentations.parquet": "53494a1267e08104ddfd5a88fac524ad",
    "xenium_nsclc/figure_4_area_sample.parquet": "c2cee784c27b1aabbac587e3a6e5b8ef",
    "xenium_nsclc/figure_4_cell_types.parquet": "5c1a624ad0c811d1f0cbd34142688ace",
    "xenium_nsclc/benchmark_scores.tsv": "8938a747f88821cd628f929928ac969e",
    "xenium_breast/xenium/experiment.xenium": "e5f855e6e2c37579e01de3e6c7a9e026",
    "xenium_breast/xenium/gene_panel.json": "8e190616dfdf2b768aeab354490666a1",
    "xenium_breast/xenium/metrics_summary.csv": "c937e7a5761b430b4c3643b0ed2aae85",
    "xenium_breast/xenium/transcripts.parquet": "30e7cbb37a9e4ab31f6d9dc28420712e",
    "xenium_breast/xenium/cells.parquet": "8397bd076436d0d0c982db917438c03b",
    "xenium_breast/xenium/cell_boundaries.parquet": "b2006704ecf26b56d1d2f992c613bad9",
    "xenium_breast/xenium/nucleus_boundaries.parquet": "e73ff46550c53ff979eac4c7200faed3",
    "xenium_breast/xenium/cell_feature_matrix.h5": "f254faa26200511463b1dc3b2e7b5c87",
    "xenium_breast/segger/segger_segmentation.parquet": "8127e7026025c044a706d8a933cbf714",
    "xenium_breast/segmentations.parquet": "e159fb3b7a3f4ac84cedbb533b29125a",
    "xenium_breast/census_breast_reference.h5ad": "1c059746963fb5baf0fd679e699c330b",
    "atera_cervical/xenium/experiment.xenium": "40db0ab68d65b2085d8b49daa5d7f9d9",
    "atera_cervical/xenium/metrics_summary.csv": "7a06c61c9b12f585d61f2d678f7bfe5e",
    "atera_cervical/xenium/transcripts.parquet": "b9fa553763b08bb37c3c72e93a1a3b7b",
    "atera_cervical/xenium/cells.parquet": "ba9e5701c2b4126074bc8d6a76dd8409",
    "atera_cervical/xenium/cell_boundaries.parquet": "adc7b8b3d812c1b298c3a48b15e6287b",
    "atera_cervical/xenium/nucleus_boundaries.parquet": "f2d746b9ab2af7ff6b529951fa091e88",
    "atera_cervical/segger/segger_segmentation.parquet": "581bc4b65f066407af5f446f392c2185",
    "atera_cervical/segger/gene_split_plan.parquet": "5eb9e5fda1050a359337d33d8b7abbf3",
    "atera_cervical/segger_gene_thresholds.tsv": "016460da3d5bc221eef6efec676015b6",
    "atera_cervical/cell_types_v2.parquet": "ba0fbb238c9a0c2a4864b8a813939481",
    "atera_cervical/cell_types_v3.parquet": "61db5fdad109c0a893781fbbf569e687",
}


def _open(url: str, method: str = "GET", timeout: float = 60, **headers):
    # The 10x Genomics CDN rejects urllib's default User-Agent.
    headers.setdefault("User-Agent", "sg_utils (https://github.com/dpeerlab/segger-analysis)")
    request = urllib.request.Request(url, method=method, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def _retry(call, what: str, retries: int = 8):
    """Run ``call`` until it succeeds; network errors are retried, HTTP 4xx errors are not."""
    for attempt in range(retries):
        try:
            return call()
        except urllib.error.HTTPError as err:
            if 400 <= err.code < 500 and err.code not in (408, 429):
                raise
            error = err
        except (OSError, http.client.HTTPException) as err:
            error = err
        if attempt < retries - 1:
            time.sleep(min(2**attempt, 30))
    raise OSError(f"{what} failed {retries} times") from error


def _read_range(url: str, start: int, end: int) -> bytes:
    """Bytes ``start`` to ``end`` (inclusive) of a remote file.

    CDNs sometimes answer a range request with the whole file (HTTP 200) when the
    object is not cached at the edge; such responses are retried like network errors.
    """

    def call():
        with _open(url, Range=f"bytes={start}-{end}") as r:
            if r.status != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                raise OSError(f"{url} ignored the range request (HTTP {r.status})")
            data = r.read()
        if len(data) != end - start + 1:
            raise OSError(f"short read from {url}: {len(data)} of {end - start + 1} bytes")
        return data

    return _retry(call, f"range request for bytes {start}-{end} of {url}")


class _RangeReader(io.RawIOBase):
    """Seekable read-only view of a remote file, served by HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        headers = _retry(lambda: _head(url), f"HEAD {url}")
        if headers.get("Accept-Ranges", "bytes") == "none":
            raise OSError(f"{url} does not support range requests")
        self.size = int(headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        data = _read_range(self.url, self.pos, end)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _head(url: str):
    with _open(url, method="HEAD") as r:
        return r.headers


def _write(src, path: Path, md5: str | None = None, size: int | None = None) -> None:
    """Copy a stream to ``path``; the file only appears once its size and md5 are verified."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    digest = hashlib.md5()
    try:
        with open(partial, "wb") as dst:
            while block := src.read(1 << 24):
                digest.update(block)
                dst.write(block)
        _verify(partial, digest.hexdigest(), md5, size)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(path)


def _verify(path: Path, found: str, md5: str | None, size: int | None) -> None:
    if size is not None and path.stat().st_size != size:
        raise OSError(f"incomplete download of {path.name.removesuffix('.part')}: {path.stat().st_size} of {size} bytes")
    if md5 is not None and found != md5:
        raise OSError(f"md5 mismatch for {path.name.removesuffix('.part')}: expected {md5}, found {found}")


def _download(url: str, path: Path, md5: str | None = None) -> None:
    """Download ``url`` to ``path``, resuming the partial file after a network error."""
    size = int(_retry(lambda: _head(url), f"HEAD {url}")["Content-Length"])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.unlink(missing_ok=True)

    def resume():
        done = partial.stat().st_size if partial.exists() else 0
        if done < size:
            with _open(url, timeout=300, Range=f"bytes={done}-") as r:
                # A server that ignores the range sends the whole file again.
                with open(partial, "ab" if r.status == 206 else "wb") as dst:
                    shutil.copyfileobj(r, dst, length=1 << 24)
        if partial.stat().st_size != size:
            raise OSError(f"incomplete download of {path.name}")

    try:
        _retry(resume, f"download of {url}")
        _verify(partial, _md5(partial) if md5 else "", md5, size)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(path)


def _md5(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _fetch_members(remote: Remote, root: Path, key: str) -> None:
    todo = [m for m in remote.members if not (root / m).exists()]
    if not todo:
        return
    root.mkdir(parents=True, exist_ok=True)
    md5 = {m: MD5.get(posixpath.normpath(f"{key}/{m}")) for m in todo}
    if remote.url.endswith("/"):  # a directory of individual files
        for member in todo:
            print(f"downloading {remote.url}{member}")
            _download(remote.url + member, root / member, md5[member])
        return
    with zipfile.ZipFile(io.BufferedReader(_RangeReader(remote.url), buffer_size=1 << 24)) as zf:
        for member in todo:
            print(f"extracting {member}")
            with zf.open(member) as src:
                _write(src, root / member, md5[member], zf.getinfo(member).file_size)


def fetch(
    dataset: str,
    data_dir: str | Path = "data",
    files: list[str] | None = None,
) -> dict[str, Path]:
    """Download the files of a dataset, skipping those already present.

    Parameters
    ----------
    dataset
        Key of :data:`DATASETS`, e.g. ``"xenium_crc"``.
    data_dir
        Root directory; files go to ``data_dir/<dataset>/``.
    files
        Subset of the dataset's files, e.g. ``["xenium", "reference"]``.

    Returns
    -------
    Local path of every requested file (a directory for archives).
    """
    entry = DATASETS[dataset]
    unknown = set(files or []) - set(entry.files)
    if unknown:
        raise KeyError(f"{dataset} has no files {sorted(unknown)}; available: {list(entry.files)}")
    root = Path(data_dir) / dataset
    paths = {}
    for name, remote in entry.files.items():
        if files is not None and name not in files:
            continue
        target = root / remote.target
        key = posixpath.normpath(f"{dataset}/{remote.target}")
        if remote.members is not None:
            _fetch_members(remote, target, key)
        elif not target.exists():
            print(f"downloading {remote.url}")
            _download(remote.url, target, MD5.get(key))
        paths[name] = target
    return paths
