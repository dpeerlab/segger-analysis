# Paper notebooks

Notebooks reproducing the figures of Heidari, Moorman et al., *Segger: fast and accurate cell segmentation of imaging-based spatial transcriptomics data*.
Each notebook starts from the public data, optionally runs Segger from the command line, and draws the panels of one figure.
Panels are written to `results/<figure>/panel_<letter>.{pdf,png}`; the figures themselves are assembled from these panels.
The Fig. 2 and Fig. 4 notebooks also write the transcripts, the cell x gene tables and the cell outlines of all segmentations to a SpatialData Zarr store (`results/<figure>/<dataset>.zarr`), with the element names of `segger export spatialdata`, for SOPA and napari-spatialdata.

| Notebook | Panels | Dataset |
|---|---|---|
| [`figure_2_xenium_crc.ipynb`](figure_2_xenium_crc.ipynb) | Fig. 2a–d | Xenium colorectal cancer with multimodal cell segmentation staining (10x Genomics) |
| [`figure_4_xenium_nsclc.ipynb`](figure_4_xenium_nsclc.ipynb) | Fig. 4a–f | Xenium non-small-cell lung cancer with NaK-ATPase membrane staining (this study) |
| [`contamination_xenium_breast.ipynb`](contamination_xenium_breast.ipynb) | Fig. 2 analysis | Xenium Prime 5K breast cancer (10x Genomics) |
| [`figure_5_xenium_healthy_colon.ipynb`](figure_5_xenium_healthy_colon.ipynb) | Fig. 5a–j | Xenium healthy human colon (10x Genomics) |
| [`ed_1_xenium_healthy_colon.ipynb`](ed_1_xenium_healthy_colon.ipynb) | Extended Data Fig. 1c,d | Xenium healthy human colon |
| [`ed_2_tiling.ipynb`](ed_2_tiling.ipynb) | Extended Data Fig. 2a–e | Six sections (Xenium, Xenium Prime, CosMx) and public dataset summaries |
| [`ed_6_xenium_healthy_colon.ipynb`](ed_6_xenium_healthy_colon.ipynb) | Extended Data Fig. 6a–j | Xenium healthy human colon |
| [`supp_1_xenium_multimodal.ipynb`](supp_1_xenium_multimodal.ipynb) | Supplementary Fig. 1a–c | Three Xenium sections with multimodal cell segmentation staining |
| [`supp_2_partitioning.ipynb`](supp_2_partitioning.ipynb) | Supplementary Fig. 2b–e | Four Xenium sections |
| [`supp_atera_cervical.ipynb`](supp_atera_cervical.ipynb) | Supplementary Fig. 10a–c | Atera whole-transcriptome cervical cancer (10x Genomics) |
| [`supp_11_xenium_nsclc.ipynb`](supp_11_xenium_nsclc.ipynb) | Supplementary Fig. 11a–i | Xenium non-small-cell lung cancer with NaK-ATPase membrane staining (this study) |

Run `figure_5_xenium_healthy_colon.ipynb` before Extended Data Fig. 1 and 6, which read its cell x gene tables from `results/figure_5`.

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

The Fig. 5, Extended Data Fig. 1, 2, 6 and Supplementary Fig. 1, 2, 11 notebooks need a CUDA 12 GPU (Linux) for RAPIDS, PyTorch, Cellpose-SAM and segger (commit `98765aa` of `main`), installed by the `gpu` extra:

```bash
uv sync --extra gpu
```

Their expensive intermediate results (cell x gene tables, HotSpot models, Cellpose masks, neighbour graphs, tile statistics) are cached in `results/<figure>` and reused when present.

## Data

`sg_utils.datasets.fetch(dataset, "data")` downloads each file once into `data/<dataset>/` and checks its size and md5.
From the 10x Genomics bundles only the files needed are read, by HTTP range requests; interrupted downloads resume.

| Dataset | Source |
|---|---|
| `xenium_crc` | [10x Genomics, Human Colon Preview Data](https://www.10xgenomics.com/datasets/human-colon-preview-data-xenium-human-colon-gene-expression-panel-1-standard), section `Human_Colon_Cancer_P1_CRC_Add_on` |
| `xenium_nsclc` | Generated for this study, `s3://dp-lab-data-public/segger/xenium_nsclc.tar.gz` |
| `xenium_breast` | [10x Genomics, Xenium Prime FFPE Human Breast Cancer](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer) |
| `atera_cervical` | 10x Genomics, Atera WTA Preview FFPE Human Cervical Cancer (`WTA_Preview_FFPE_Cervical_Cancer_outs.zip`) |
| `xenium_healthy_colon` | [10x Genomics, Human Colon Preview Data](https://www.10xgenomics.com/datasets/human-colon-preview-data-xenium-human-colon-gene-expression-panel-1-standard), non-diseased section with add-on panel |

Four sections used only for the scaling and segmentation-agreement panels are not downloaded by `sg_utils.datasets`; the notebooks expect their output bundles in `data/xenium_lung_multimodal/xenium` ([Xenium lung cancer, multimodal cell segmentation](https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard)), `data/xenium_healthy_colon_base/xenium` (non-diseased colon, base panel, from the Human Colon Preview Data), `data/xenium_mouse_liver/xenium` ([Xenium v1 mouse liver](https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_mouse_liver_healthy_v1_small.tar.gz)) and `data/cosmx_pancreas/cosmx` ([CosMx human pancreas](https://brukerspatialbiology.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/cosmx-smi-human-pancreas-ffpe-dataset/)).

Segmentations of all methods, the scRNA-seq references, the CellTypist models and the colon marker genes are hosted under `https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/paper/`.

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
