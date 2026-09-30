# Paper notebooks

Notebooks reproducing the figures of Heidari, Moorman et al., *Segger: fast and accurate cell segmentation of imaging-based spatial transcriptomics data*.
Each notebook starts from the public data, optionally runs Segger from the command line, and draws the panels of one figure.
Panels are written to `results/<figure>/panel_<letter>.{pdf,png}`; the figures themselves are assembled from these panels.
The Xenium notebooks also write the transcripts, the cell x gene tables and the cell outlines of all segmentations to a SpatialData Zarr store (`results/<figure>/<dataset>.zarr`), with the element names of `segger export spatialdata`, for SOPA and napari-spatialdata.

| Notebook | Panels | Dataset |
|---|---|---|
| [`figure_2_xenium_crc.ipynb`](figure_2_xenium_crc.ipynb) | Fig. 2a–d | Xenium colorectal cancer with multimodal cell segmentation staining (10x Genomics) |
| [`figure_4_xenium_nsclc.ipynb`](figure_4_xenium_nsclc.ipynb) | Fig. 4a–f | Xenium non-small-cell lung cancer with NaK-ATPase membrane staining (this study) |
| [`contamination_xenium_breast.ipynb`](contamination_xenium_breast.ipynb) | Fig. 2 analysis | Xenium Prime 5K breast cancer (10x Genomics) |
| [`supp_atera_cervical.ipynb`](supp_atera_cervical.ipynb) | Supplementary Fig. 10a–c | Atera whole-transcriptome cervical cancer (10x Genomics) |

All plotting and analysis functions live in `sg_utils` (`src/sg_utils`): `sg_utils.datasets` downloads the data, `sg_utils.io` reads and writes it, `sg_utils.tl` holds cell typing, segmentation metrics, cell outlines and neighbourhood graphs, and `sg_utils.pl` the panels.

## Environment

The notebooks run on a CPU with 24 GB of memory; the Atera notebook reads its 10⁹ transcripts in blocks and needs about 2 h. Install the pinned environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/dpeerlab/segger-analysis
cd segger-analysis
uv sync            # Python 3.11, versions from uv.lock
uv run jupyter lab
```

Running Segger itself (`RUN_SEGGER = True` in a notebook) needs a CUDA GPU and the `segger` command on the kernel's path, installed as described in the [segger documentation](https://segger-segmentation.readthedocs.io/en/latest/installation.html); the Atera notebook needs commit `8ee343c` of its `integration/all` branch.
Without it, the notebooks download the segmentation used in the paper.

## Data

`sg_utils.datasets.fetch(dataset, "data")` downloads each file once into `data/<dataset>/` and checks its size and md5.
From the 10x Genomics bundles only the files needed are read, by HTTP range requests; interrupted downloads resume.

| Dataset | Source |
|---|---|
| `xenium_crc` | [10x Genomics, Human Colon Preview Data](https://www.10xgenomics.com/datasets/human-colon-preview-data-xenium-human-colon-gene-expression-panel-1-standard), section `Human_Colon_Cancer_P1_CRC_Add_on` |
| `xenium_nsclc` | Generated for this study, `s3://dp-lab-data-public/segger/xenium_nsclc.tar.gz` |
| `xenium_breast` | [10x Genomics, Xenium Prime FFPE Human Breast Cancer](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer) |
| `atera_cervical` | 10x Genomics, Atera WTA Preview FFPE Human Cervical Cancer (`WTA_Preview_FFPE_Cervical_Cancer_outs.zip`) |

Segmentations of all methods, the scRNA-seq references and the CellTypist models are hosted under `https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/paper/`.

## Paper-only inputs

A few panels depend on a random draw or on results of other analyses of the paper. These inputs are downloaded and used when their flag in the `parameters` cell is `True`, the default in these notebooks. The same notebooks, with every flag set to `False` and `RUN_SEGGER = True`, are the tutorials of the segger documentation.

| Flag | Notebook | Input |
|---|---|---|
| `PAPER_BENCHMARK` | Fig. 2, Fig. 4 | Benchmark scores of Supplementary Fig. 3.2 (`segger validate`, on the `integration/all` branch of segger, run on the benchmark segmentations), drawn for coverage and spurious co-expression in Fig. 2d and for spurious co-expression and PMR in Fig. 4a |
| `PAPER_SAMPLE` | Fig. 4 | The 6,000 random cells per method sampled for the cell areas of Fig. 4c |
| `PAPER_THRESHOLDS` | Atera | The per-gene similarity thresholds tabulated for Supplementary Fig. 10 |
| `PAPER_LABELS` | Fig. 4 | The compartments and cell types of the Cellpose and Baysor cells, typed for the paper when their cell tables were built |
| `PAPER_LABELS` | Atera | The display cell types of Supplementary Fig. 10a,b |

The Fig. 2, Fig. 4 and Atera notebooks include a table of the values printed in the paper next to the values the notebook computed.

## Running a notebook non-interactively

The first code cell after the imports is tagged `parameters`, so notebooks can be executed with [papermill](https://papermill.readthedocs.io) or nbconvert:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace paper/figure_2_xenium_crc.ipynb
```
