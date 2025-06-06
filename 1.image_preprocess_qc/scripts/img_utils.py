import matplotlib as mpl
import re

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


## Define mapping between simple names and folder names
batch_dict = {
    "B7A1R1": "2024_01_23_Batch_7",
    "B7A2R1": "2024_01_23_Batch_7",
    "B8A1R2": "2024_02_06_Batch_8",
    "B8A2R2": "2024_02_06_Batch_8",
    "B13A7A8P1": "2025_01_27_Batch_13",
    "B13A7A8P2": "2025_01_27_Batch_13",
    "B14A7A8P1": "2025_01_28_Batch_14",
    "B14A7A8P2": "2025_01_28_Batch_14",
    "B15A1A2P1": "2025_03_17_Batch_15",
    "B16A1A2P1": "2025_03_17_Batch_16",
}


## Store a large dict for mapping platemaps to img measurements on cell-painting gallery
plate_dict = {
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
        cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#65fe08"])
    elif channel == "DAPI":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#0000FF"])
    elif channel == "Mito":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#FF0000"]) 
    elif channel == "AGP":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#FFFF00"]) 
    else:
        cmap = "gray"
    return cmap


# Convert letter rows to numbers
def well_to_coordinates(well):
    row_letter, col_number = re.match(r"([A-P])(\d{2})", well).groups()
    row_index = ord(row_letter) - ord('A') + 1  # Convert 'A'->1, 'B'->2, ..., 'P'->16
    col_index = int(col_number)  # Convert string column to integer
    return well, row_index, col_index


# Compute distances from edges and find the most centered well
def compute_distance(row, col):
    return min(row - 1, 16 - row, col - 1, 24 - col)  # Distance from nearest edge