major_colors = {
    "Cancer Epithelial": "#e77377",
    "Normal Epithelial": "#8bb7f4",
    "T-cells": "#84d68e",
    "B-cells": "#d8f55e",
    "CAFs": "#B999E5",
    "Plasmablasts": "#f3fccc",
    "Myeloid": "#E1D168",
    "Endothelial": "#dcc8c7",
    "PVL": "#eadedd",
}

extended_colors = {
    "Cancer Epithelial": "#d14343",  # More vibrant red
    "Normal Epithelial": "#5a96e8",  # More vibrant blue
    "T-cells": "#43a853",           # Brighter green
    "B-cells": "#c4d400",           # Vibrant yellow-green
    "CAFs": "#9966cc",              # Brighter purple
    "Plasmablasts": "#eefc79",      # Brighter pale yellow
    "Myeloid": "#d6b63a",           # Richer golden yellow
    "Endothelial": "#b79f9f",       # Warmer beige
    "PVL": "#d4c1c1",               # Slightly richer off-white
    "Adipocytes": "#c86d85",        # Brighter coral
    "Breast glandular cells": "#82caff",  # Vibrant light blue
    "Dendritic cells": "#f7b362",        # Brighter yellow-orange
    "Immune cells": "#9bcc72",           # Brighter green
    "Macrophages": "#e5d78f",            # Rich tan
    "Mast cells": "#99c9f0",             # Brighter purple-blue
    "Monocytes": "#d5f572",              # Brighter lime
    "NK cells": "#74c683",               # Vibrant teal
    "Neutrophils": "#f48365",            # Brighter peach
    "Smooth muscle cells": "#bd92af",    # Richer mauve
    "Epithelial cells": "#9966cc",       # Brighter version of CAFs
    "Other": "#9e9e9e",                  # Neutral gray with more contrast
}



lookup_markers = {
    'Adipocytes'                : 'Adipocytes',
    'B cells'                   : 'B-cells',
    'Breast cancer'             : 'Cancer Epithelial',
    'Breast glandular cells'    : 'Breast glandular cells',
    'Breast myoepithelial cells': 'Normal Epithelial',
    'Epithelial cells'          : 'Normal Epithelial',
    'Fibroblasts'               : 'CAFs',
    'T cells'                   : 'T-cells',
    'Myeloid cells'             : 'Myeloid',
    'Dendritic cells'           : 'Dendritic cells',
    'Endothelial cells'         : 'Endothelial',
    'Immune cells'              : 'Immune cells',
    'Macrophages'               : 'Macrophages',
    'Mast cells'                : 'Mast cells',
    'Monocytes'                 : 'Monocytes',
    'NK cells'                  : 'NK cells',
    'Neutrophils'               : 'Neutrophils',
    'Plasma cells'              : 'Plasma cells',
    'Smooth muscle cells'       : 'Smooth muscle cells',
    'Plasma cells'              : 'Plasmablasts',
    'Custom'                    : 'Other'
}

method_colors = {
    "segger": "#C72228",
    "segger+": "#E69F00",
    "segger++": "#F0E442",
    "segger_embedding": "#C72228",
    "segger (nucleus+)": '#f48365',
    "segger (nucleus-)": '#E69F00',
    "segger_fragments": '#E69F00',
    "Baysor": "#72A8E6",
    "Baysor (nucleus+)": "#0F4A9C",
    "Baysor (nucleus-)": "#0072B2",
    "10x_Cell": "#BB8BEB",
    "10x_Nucleus": "#F46CB7",
    'BIDCell': '#CDF3AF'
}