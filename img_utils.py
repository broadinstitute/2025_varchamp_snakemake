import re
import pickle
import numpy as np
from functools import reduce
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

## allele collection direcotory
# ALLELE_COLLECT_DIR = "../../../../../../../1_allele_collection"

## snakemake pipeline directory
SNAKEMAKE_PIPELINE_DIR = "/home/shenrunx/igvf/varchamp/2025_varchamp_snakemake"

## .tiff img directory, downloaded from aws cell-painting gallery
TIFF_IMGS_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/1.image_preprocess_qc/inputs/cpg_imgs"

## meta platemap directory, stores the allele location on each well and plate
PLATEMAP_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/2.snakemake_pipeline/inputs/metadata/platemaps/" + "{batch_id}/platemap"

## IMG QC related directories
IMGS_QC_BG_SUM_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/1.image_preprocess_qc/outputs/plate_bg_summary"
IMGS_QC_METRICS_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/inputs/1.plate_well_qc_metrics"

IMG_METADATA_FILE = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/outputs/0.img_metadata_qc/allele_meta_df.csv"
IMG_METADATA_DICT_FILE = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/outputs/0.img_metadata_qc/allele_meta_df_dict.pckl"
IMG_QC_SUM_DF_FILE = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/outputs/0.img_metadata_qc/img_well_qc_sum_df.csv"
IMG_QC_SUM_DICT_FILE = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/outputs/0.img_metadata_qc/img_well_qc_sum_dict.pckl"

## batch profiles directory
PROF_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/2.snakemake_pipeline/outputs/batch_profiles"

## cell count and abundance change directory
CC_ABUND_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/outputs/1.cell_count_abundance_change"

## classification results directory from snakemake pipeline
CLASS_ANALYSES_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/2.snakemake_pipeline/outputs/classification_analyses"
CLASS_RESULTS_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/2.snakemake_pipeline/outputs/classification_results"

## classification results summary from snakemake pipeline's results
CLASS_INTERIM_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/inputs/2.classification_results"
CLASS_SUMMARY_DIR = f"{SNAKEMAKE_PIPELINE_DIR}/3.downstream_analyses/outputs/2.classification_results"

## CellProfiler Feature Sets
FEAT_SETS = ["GFP", "DNA", "Mito", "AGP", "Morph"]

BATCH_LIST_DICT = {
    "2024_01_23_Batch_7": "2024_02_Batch_7-8", 
    "2024_02_06_Batch_8": "2024_02_Batch_7-8",
    "2024_12_09_Batch_11": "2024_12_Batch_11-12", 
    "2024_12_09_Batch_12": "2024_12_Batch_11-12",
    "2025_01_27_Batch_13": "2025_01_Batch_13-14", 
    "2025_01_28_Batch_14": "2025_01_Batch_13-14",
    "2025_03_17_Batch_15": "2025_03_Batch_15-16", 
    "2025_03_17_Batch_16": "2025_03_Batch_15-16",
    "2025_06_10_Batch_18": "2025_06_Batch_18-19",
    "2025_06_10_Batch_19": "2025_06_Batch_18-19"
}


BIO_REP_BATCHES_DICT = {
    "2024_01_Batch_7-8": ("2024_01_23_Batch_7", "2024_02_06_Batch_8"),
    "2024_12_Batch_11-12": ("2024_12_09_Batch_11", "2024_12_09_Batch_12"),
    "2025_03_Batch_15-16": ("2025_03_17_Batch_15", "2025_03_17_Batch_16"),
    "2025_01_Batch_13-14": ("2025_01_27_Batch_13", "2025_01_28_Batch_14"),
    "2025_06_Batch_18-19": ("2025_06_10_Batch_18", "2025_06_10_Batch_19")
}


BIO_BATCH_MAP_DICT = {
    "B7": "B_7-8",
    "B8": "B_7-8",
    "B11": "B_11-12",
    "B12": "B_11-12",
    "B15": "B_15-16",
    "B16": "B_15-16",
    "B13": "B_13-14",
    "B14": "B_13-14",
    "B18": "B_18-19",
    "B19": "B_18-19",
}


## Letter dict to convert well position to img coordinates
letter_dict = {
    "A": "01",
    "B": "02",
    "C": "03",
    "D": "04",
    "E": "05",
    "F": "06",
    "G": "07",
    "H": "08",
    "I": "09",
    "J": "10",
    "K": "11",
    "L": "12",
    "M": "13",
    "N": "14",
    "O": "15",
    "P": "16",
}


## Channel dict to map channel to cellular compartments
channel_dict = {
    "DAPI": "1",
    "GFP": "2",
    "AGP": "3",
    "Mito": "4",
    "Brightfield1": "5",
    "Brightfield2": "6",
    "Brightfield": "7",
}

letter_dict_rev = {v: k for k, v in letter_dict.items()}
channel_dict_rev = {v: k for k, v in channel_dict.items()}
channel_list = list(channel_dict_rev.values())[:-3]

## Define mapping between simple names and folder names
batch_dict = {
    "B7A1R1": "2024_01_23_Batch_7",
    "B7A2R1": "2024_01_23_Batch_7",
    "B8A1R2": "2024_02_06_Batch_8",
    "B8A2R2": "2024_02_06_Batch_8",
    "B11A1R1": "2024_12_09_Batch_11",
    "B12A1R2": "2024_12_09_Batch_12",
    "B13A7A8P1": "2025_01_27_Batch_13",
    "B13A7A8P2": "2025_01_27_Batch_13",
    "B14A7A8P1": "2025_01_28_Batch_14",
    "B14A7A8P2": "2025_01_28_Batch_14",
    "B15A1A2P1": "2025_03_17_Batch_15",
    "B16A1A2P1": "2025_03_17_Batch_16",
    "B18A8A10R1": "2025_06_10_Batch_18",
    "B19A8A10R1": "2025_06_10_Batch_19"
}


## color map to plot different types of alleles on a platemap
color_map = {
    'TC': 'slategrey', # Grey for controls
    'NC': 'gainsboro', 
    'PC': 'plum',
    'cPC': 'pink',
    'cNC': 'lightgrey',
    'allele': 'salmon',  # Tomato for disease
    'disease_wt': 'lightskyblue',  # Skyblue for reference
    '': 'white'  # White for missing wells
}


## Store a large dict for mapping platemaps to img measurements on cell-painting gallery
plate_dict = {
    ## Batch 18
    "B18A8A10R1_P1": {
        "T1": '2025_05_28_B18A8A10R1_P1T1__2025-05-28T08_23_18-Measurement1',
        "T2": '2025_05_28_B18A8A10R1_P1T2__2025-05-28T09_32_29-Measurement1',
        "T3": '2025_05_28_B18A8A10R1_P1T3__2025-05-28T10_50_59-Measurement1',
        "T4": '2025_05_28_B18A8A10R1_P1T4__2025-05-28T12_00_22-Measurement1'
    },

    "B18A8A10R1_P2": {
        "T1": '2025_06_02_B18A8A10R1_P2T1__2025-06-02T08_19_26-Measurement1',
        "T2": '2025_06_02_B18A8A10R1_P2T2__2025-06-02T09_29_48-Measurement1',
        "T3": '2025_06_02_B18A8A10R1_P2T3__2025-06-02T11_08_44-Measurement1',
        "T4": '2025_06_02_B18A8A10R1_P2T4__2025-06-02T12_18_34-Measurement1'
    },
    
    ## Batch 19
    "B19A8A10R1_P1": {
        "T1": '2025_06_02_B19A8A10R1_P1T1__2025-06-02T14_07_21-Measurement1',
        "T2": '2025_06_03_B19A8A10R1_P1T2__2025-06-03T08_16_43-Measurement2',
        "T3": '2025_06_03_B19A8A10R1_P1T3__2025-06-03T09_40_25-Measurement1',
        "T4": '2025_06_03_B19A8A10R1_P1T4__2025-06-03T11_55_37-Measurement1'
    },

    "B19A8A10R1_P2": {
        "T1": '2025_06_03_B19A8A10R1_P2T1__2025-06-03T13_27_13-Measurement1',
        "T2": '2025_06_04_B19A8A10R1_P2T2__2025-06-04T08_17_36-Measurement2',
        "T3": '2025_06_04_B19A8A10R1_P2T3__2025-06-04T09_30_37-Measurement1',
        "T4": '2025_06_04_B19A8A10R1_P2T4__2025-06-04T11_49_06-Measurement1'
    },
    
    ## Batch 15
    "B15A1A2P1_R1": {
        "T1": '2025-03-17_B15A1A2_P1T1__2025-03-17T08_34_13-Measurement_1',
        "T2": '2025-03-17_B15A1A2_P1T3__2025-03-17T11_08_17-Measurement_2',
        "T3": '2025-03-17_B15A1A2_P1T2__2025-03-17T09_39_24-Measurement_1',
        "T4": '2025-03-17_B15A1A2_P1T4__2025-03-17T12_28_48-Measurement_1'
    },
    
    ## Batch 16
    "B16A1A2P1_R2": {
        "T1": '2025-03-18_B16A1A2_P1T2__2025-03-18T08_45_49-Measurement_1',
        "T2": '2025-03-18_B16A1A2_P1T3__2025-03-18T10_08_59-Measurement_1',
        "T3": '2025-03-17_B16A1A2_P1T1__2025-03-17T13_48_08-Measurement_1',
        "T4": '2025-03-18_B16A1A2_P1T4__2025-03-18T11_18_31-Measurement_1'
    },

    ## Batch 13
    "B13A7A8P1_R1": {
        "T1": "2025_01_27_B13A7A8P1_T1__2025_01_27T08_46_50_Measurement_1",
        "T2": "2025_01_27_B13A7A8P1_T2__2025_01_27T09_53_48_Measurement_1",
        "T3": "2025_01_27_B13A7A8P1_T3__2025_01_27T11_10_53_Measurement_1",
        "T4": "2025_01_27_B13A7A8P1_T4__2025_01_27T12_17_23_Measurement_1",
    },

    "B13A7A8P2_R1": {
        "T1": "2025_01_27_B13A7A8P2_T1__2025_01_27T13_39_08_Measurement_1",
        "T2": "2025_01_27_B13A7A8P2_T2__2025_01_27T15_06_52_Measurement_1",
        "T3": "2025_01_27_B13A7A8P2_T3__2025_01_27T16_24_23_Measurement_1",
        "T4": "2025_01_27_B13A7A8P2_T4__2025_01_27T17_34_19_Measurement_1",
    },

    ## Batch 14
    "B14A7A8P1_R2": {
        "T1": '2025_01_28_B14A7A8P1_T1__2025_01_28T08_50_18_Measurement_1',
        "T2": '2025_01_28_B14A7A8P1_T2__2025_01_28T09_53_20_Measurement_1',
        "T3": '2025_01_28_B14A7A8P1_T3__2025_01_28T12_28_32_Measurement_3',
        "T4": '2025_01_28_B14A7A8P1_T4__2025_01_28T11_14_00_Measurement_2',
    },

    "B14A7A8P2_R2": {
        "T1": '2025_01_28_B14A7A8P2_T1__2025_01_28T13_32_03_Measurement_1',
        "T4": '2025_01_28_B14A7A8P2_T4__2025_01_28T17_11_31_Measurement_1',
        "T2": '2025_01_28_B14A7A8P2_T2__2025_01_28T14_42_58_Measurement_1',
        "T3": '2025_01_28_B14A7A8P2_T3__2025_01_28T16_06_55_Measurement_1'
    },

    ## Batch 11
    "B11A1R1": "2024-12-09_B11A1R1__2024-12-09T08_49_55-Measurement_1",

    ## Batch 12
    "B12A1R2": "2024-12-09_B12A1R2__2024-12-09T10_34_39-Measurement_1", 

    ## Batch 7
    "B7A1R1_P1": {"T1": "2024_01_17_B7A1R1_P1T1__2024_01_17T08_35_58_Measurement_1",
                  "T2": "2024_01_17_B7A1R1_P1T2__2024_01_17T10_13_45_Measurement_1",
                  "T3": "2024_01_17_B7A1R1_P1T3__2024_01_17T11_58_08_Measurement_1",
                  "T4": "2024_01_17_B7A1R1_P1T4__2024_01_17T13_45_14_Measurement_1"},
    
    "B7A1R1_P2": {"T1": "2024_01_17_B7A1R1_P2T1__2024_01_17T15_33_09_Measurement_1",
                  "T2": "2024_01_17_B7A1R1_P2T2__2024_01_18T08_25_01_Measurement_1",
                  "T3": "2024_01_17_B7A1R1_P2T3__2024_01_18T10_47_36_Measurement_1",
                  "T4": "2024_01_17_B7A1R1_P2T4__2024_01_18T12_48_20_Measurement_1"},
        
    "B7A1R1_P3": {"T1": "2024_01_18_B7A1R1_P3T1__2024_01_18T14_27_08_Measurement_1",
                  "T2": "2024_01_19_B7A1R1_P3T2__2024_01_19T08_23_30_Measurement_1",
                  "T3": "2024_01_19_B7A1R1_P3T3__2024_01_19T10_01_45_Measurement_1",
                  "T4": "2024_01_19_B7A1R1_P3T4__2024_01_19T12_00_10_Measurement_1"},
            
    "B7A1R1_P4": {"T1": "2024_01_19_B7A1R1_P4T1__2024_01_19T13_50_55_Measurement_1",
                  "T2": "2024_01_23_B7A1R1_P4T2__2024_01_23T10_13_00_Measurement_1",
                  "T3": "2024_01_22_B7A1R1_P4T3__2024_01_22T08_37_41_Measurement_1",
                  "T4": "2024_01_22_B7A1R1_P4T4__2024_01_22T10_27_16_Measurement_1"},
    
    ## Batch 8
    "B8A1R2_P1": {"T1": "2024_01_31_B8A1R2_P1T1__2024_01_31T10_11_57_Measurement_1",
                  "T2": "2024_01_31_B8A1R2_P1T2__2024_01_31T08_35_51_Measurement_2",
                  "T3": "2024_01_31_B8A1R2_P1T3__2024_01_31T12_09_14_Measurement_1",
                  "T4": "2024_01_31_B8A1R2_P1T4__2024_01_31T14_02_18_Measurement_2"},

    "B8A1R2_P2": {"T1": "2024_01_31_B8A1R2_P2T1__2024_01_31T15_41_23_Measurement_1",
                  "T2": "2024_02_01_B8A1R2_P2T2__2024_02_01T10_23_20_Measurement_2",
                  "T3": "2024_02_01_B8A1R2_P2T3__2024_02_01T12_16_30_Measurement_4",
                  "T4": "2024_02_01_B8A1R2_P2T4__2024_02_01T14_05_52_Measurement_1"},
        
    "B8A1R2_P3": {"T1": "2024_02_02_B8A1R2_P3T1__2024_02_02T08_32_30_Measurement_2",
                  "T2": "2024_02_02_B8A1R2_P3T2__2024_02_02T10_08_05_Measurement_1",
                  "T3": "2024_02_02_B8A1R2_P3T3__2024_02_02T11_58_46_Measurement_2",
                  "T4": "2024_02_02_B8A1R2_P3T4__2024_02_02T13_51_50_Measurement_1"},
            
    "B8A1R2_P4": {"T1": "2024_02_02_B8A1R2_P4T1__2024_02_02T15_32_28_Measurement_1",
                  "T2": "2024_02_05_B8A1R2_P4T2__2024_02_05T08_22_47_Measurement_2",
                  "T3": "2024_02_05_B8A1R2_P4T3__2024_02_05T10_00_30_Measurement_1",
                  "T4": "2024_02_05_B8A1R2_P4T4__2024_02_05T11_38_50_Measurement_1"},
}


def channel_to_cmap(channel):
    if channel == "GFP":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("gfp_cmap", ["#000","#65fe08"])
    elif channel == "DAPI":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("dapi_cmap", ["#000","#0000FF"])
    elif channel == "Mito":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("mito_cmap", ["#000","#FF0000"])
    elif channel == "AGP":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("agp_cmap", ["#000","#FFFF00"])
    else:
        cmap = "gray"
    return cmap


def channel_to_rgb(channel):
    """Convert channel name to RGB values for multi-channel visualization"""
    channel_rgb_map = {
        "GFP": [0.396, 0.996, 0.031],     # Green
        "DAPI": [0.0, 0.0, 1.0],          # Blue  
        "Mito": [1.0, 0.0, 0.0],          # Red
        "AGP": [1.0, 1.0, 0.0],           # Yellow
    }
    return channel_rgb_map.get(channel, [1.0, 1.0, 1.0])  # Default to white


# Convert letter rows to numbers
def well_to_coordinates(well):
    row_letter, col_number = re.match(r"([A-P])(\d{2})", well).groups()
    row_index = ord(row_letter) - ord('A') + 1  # Convert 'A'->1, 'B'->2, ..., 'P'->16
    col_index = int(col_number)  # Convert string column to integer
    return well, row_index, col_index


# Compute distances from edges and find the most centered well
def compute_distance(row, col):
    return min(row - 1, 16 - row, col - 1, 24 - col)  # Distance from nearest edge


def plot_platemap(
    df,
    plate_name,
    well_pos_col="well_position",
    # this is the column to color by (categorical or continuous)
    value_col="node_type",
    # these columns will be concatenated into the annotation text
    label_cols=("gene_allele",),
    value_type="categorical",   # or "continuous"
    ax=None,
    continuous_cmap="vlag",  # matplotlib colormap for continuous mode
    categorical_colors=color_map,     # dict for categorical → color
    grid_square=None
):
    # 1) build empty 16×24 grid
    rows = list("ABCDEFGHIJKLMNOP")
    cols = [f"{i:02d}" for i in range(1,25)]
    plate_grid = (
        pl.DataFrame({c: [""]*16 for c in cols})
          .with_row_index("row_index")
          .unpivot(index="row_index", on=cols, variable_name="col_label", value_name="_")
          .with_columns(
              pl.col("row_index").map_elements(lambda i: rows[i], return_dtype=pl.Utf8).alias("row_label")
          )
    )
    # display(plate_grid)

    # 2) extract row/col from your df’s well_position
    df2 = df.with_columns([
        pl.col(well_pos_col).str.head(1).alias("row_label"),
        pl.col(well_pos_col).str.slice(1).alias("col_label")
    ])

    # 3) join
    plate = plate_grid.join(df2, on=["row_label","col_label"], how="left")

    # 4) pivot out two matrices:
    #    A) data matrix for coloring
    #    B) text matrix for annotation
    # first build annotation text by concatenating label_cols
    plate = plate.with_columns(
        reduce(
            lambda acc, c: acc + "\n" + \
            pl.col(c).round(2).cast(pl.Utf8).fill_null(""),
            label_cols[1:],
            pl.col(label_cols[0]).fill_null("").str.replace_all("_", "\n")
        ).alias("_annot")
    )
    # display(plate)

    # pivot color‐matrix
    data_matrix = plate.pivot(
        index="row_label", on="col_label", values=value_col
    )

    # pivot annotation‐matrix
    annot_matrix = plate.pivot(
        index="row_label", on="col_label", values="_annot"
    ).fill_null("")

    # convert to numpy
    # drop the implicit “row_label” column in position 0
    data = data_matrix[:,1:].to_numpy()
    ann = annot_matrix[:,1:].to_numpy()

    # 5) choose coloring
    if value_type == "categorical":
        if categorical_colors is None:
            raise ValueError("Must supply categorical_colors when value_type='categorical'")
        # map each category in data to its color
        # build vectorized map
        cmap_array = np.vectorize(lambda x: categorical_colors.get(x, "white"))(data)
        # For seaborn we draw a dummy zero‐matrix
        plot_data = np.zeros_like(data, dtype=float)
        cmap = None
    else:
        # continuous: data is numeric
        plot_data = data.astype(float)
        cmap = continuous_cmap
        cmap_array = None

    # 6) plot
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(35,14))
        
    sns.heatmap(
        plot_data,
        ax=ax,
        annot=ann if value_type=="categorical" else None,
        fmt="",
        cmap=cmap,
        cbar=(value_type=="continuous"),
        # linewidths=0,
        # linecolor="white",
        square=True,
        annot_kws={"size":9, "color": "black"}
    )

    # if categorical: overlay colored rectangles
    if value_type=="categorical":
        for i in range(cmap_array.shape[0]):
            for j in range(cmap_array.shape[1]):
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    color=cmap_array[i,j],
                    # ec="black"
                ))
    else:
        # create combined annotation: value + other labels
        # you could easily extend to show gene_allele too by rebuilding ann
        for i in range(ann.shape[0]):
            for j in range(ann.shape[1]):
                txt = ann[i,j]
                # if you want gene_allele too: append "\n"+ann[i,j]
                ax.text(
                    j+0.5, i+0.5, txt,
                    ha="center", va="center", fontsize=9.5, color="black"
                )

    if grid_square is not None:
        grid_sq_mat = plate.pivot(
            index="row_label", on="col_label", values=grid_square
        )[:,1:]#.to_numpy()
        for i in range(grid_sq_mat.shape[0]):
            for j in range(grid_sq_mat.shape[1]):
                if grid_sq_mat[i,j] is not None and grid_sq_mat[i,j]>=1:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1,
                        linewidth=2, edgecolor="red", facecolor="none"
                    ))

    # 7) finalize axes
    ax.set_title(f"384-Well Plate: {plate_name}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(len(cols))+0.5)
    ax.set_xticklabels(cols, rotation=0)
    ax.set_yticks(np.arange(len(rows))+0.5)
    ax.set_yticklabels(rows, rotation=0)
    # plt.tight_layout()
    # plt.show()
    return plate