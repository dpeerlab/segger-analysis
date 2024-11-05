'''
This file contains standardized imports and variables used throughout the 
analysis notebooks for "Title"
by Heidari et al (DOI: )
'''
# Packages used throughout
from matplotlib import pyplot as plt
from pathlib import Path
import geopandas as gpd
import seaborn as sns
from tqdm import tqdm
import segger as sg
import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
import warnings
import sg_utils
import shapely
import cudf
import json
import sys
import os

# To ignore pandas warnings
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings(
    "ignore",
    "The default of observed=False is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

# Standard variables referenced throughout
base_dir = Path(__file__).parents[1]
data_dir = base_dir / 'data'
media_dir = base_dir / 'media'

# Plotting styles used throughout
assets_dir = Path(sg_utils.__path__[0]) / 'pl/assets'
plt.style.use(assets_dir / 'default.mplstyle')
with open(assets_dir / 'named_colors.json', 'r') as f:
    named_colors = json.load(f)