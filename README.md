# Segger: Reproducible Analyses for Fast and Accurate Cell Segmentation in Imaging-Based Spatial Omics

This repository contains code and analyses for reproducing figures in the preprint *Segger: Fast and accurate cell segmentation of imaging-based spatial omics data*. **Segger** is a graph-based segmentation framework that improves transcript assignment in spatial omics data while offering high sensitivity, specificity, and scalability.

## Repository Structure

- **`notebooks_share/`** – Jupyter notebooks for reproducing figures and performing key analyses.  
- **`src/`** – Python modules for data preprocessing, visualization, and utility functions.  

## Associated Data

The datasets used in this study are publicly available and can be accessed as follows:

### AWS-hosted datasets for this publication  
- **Xenium Breast Cancer Dataset:**  
  [Download (tar)](https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_breast.tar)  
- **Xenium Colon Dataset:**  
  [Download (tar.gz)](https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_colon.tar.gz)  
- **Xenium NSCLC Dataset:**  
  [Download (tar.gz)](https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_nsclc.tar.gz)  
- **MERSCOPE Brain Dataset:**  
  [Download (tar.gz)](https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/merscope_brain.tar.gz)
- **CosMx Pancreas Dataset:**  
  [Download (tar.gz)](https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/cosmx_pancreas.tar.gz)  
- **Xenium CRC Multimodal Dataset:**  
  [Download (tar.gz)](https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_colon_multimodal.tar.gz)  

### Publicly available IST datasets  
- **Xenium Breast Cancer (10x Genomics):**  
  [Dataset Link](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)  
- **Xenium Colon (10x Genomics):**  
  [Dataset Link](https://www.10xgenomics.com/datasets/human-colon-preview-data-xenium-human-colon-gene-expression-panel-1-standard)
- **MERSCOPE Brain (Vizgen):**  
  [Dataset Link](https://info.vizgen.com/access-vizgen-data-release-1000-plex)  
- **CosMx Pancreas (Nanostring):**  
  [Dataset Link](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/cosmx-smi-human-pancreas-ffpe-dataset)  
- **Xenium Colon (10x Genomics):**  
  [Dataset Link](https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc)  

**Download the datasets**
Run the following commands to download and extract the datasets:  

```sh
wget https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_breast.tar
wget https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_colon.tar.gz
wget https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_nsclc.tar.gz
wget https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/merscope_brain.tar.gz
wget https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/cosmx_pancreas.tar.gz
wget https://dp-lab-data-public.s3.us-east-1.amazonaws.com/segger/xenium_colon_multimodal.tar.gz

tar -xvf xenium_breast.tar
tar -xvzf xenium_colon.tar.gz
tar -xvzf xenium_nsclc.tar.gz
tar -xvzf merscope_brain.tar.gz
tar -xvzf cosmx_pancreas.tar.gz
tar -xvzf xenium_colon_multimodal.tar.gz
```

## Contact

For questions or issues, please open a GitHub issue or contact the authors of the manuscript.
