"""Filter cells and wells"""

import polars as pl

## Now defined in the config .json, no longer need to be redefined here anymore
# define control types
# TC = ["EGFP"]
# NC = ["RHEB", "MAPK9", "PRKACB", "SLIRP"]
# PC = ["ALK", "ALK_Arg1275Gln", "PTK2B"]
# cNC = ["Renilla"]
# cPC = [
#     "KRAS",
#     "PTK2B",
#     "GHSR",
#     "ABL1",
#     "BRD4",
#     "OPRM1",
#     "RB1",
#     "ADA",
#     "WT PMP22",
#     "LYN",
#     "TNF",
#     "CYP2A6",
#     "CSK",
#     "PAK1",
#     "ALDH2",
#     "CHRM3",
#     "KCNQ2",
#     "ALK T1151M",
#     "PRKCE",
#     "LPAR1",
#     "PLP1",
# ]


def filter_cells(input_path: str, output_path: str, TC: list, NC: list, PC: list, cPC: list)-> None:
    # filter cells based on nuclear:cellular area
    profiles_path = input_path.replace(
        "profiles_tcdropped_filtered_var_mad_outlier_featselect", "profiles"
    )

    cell_IDs = (
        pl.scan_parquet(profiles_path)
        .select([
            "Metadata_well_position",
            "Metadata_Plate",
            "Metadata_ImageNumber",
            "Metadata_ObjectNumber",
            "Nuclei_AreaShape_Area",
            "Cells_AreaShape_Area",
        ])
        .with_columns(
            (pl.col("Nuclei_AreaShape_Area") / pl.col("Cells_AreaShape_Area")).alias(
                "Nucleus_Cell_Area"
            ),
            pl.concat_str(
                [
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber",
                ],
                separator="_",
            ).alias("Metadata_CellID"),
        )
        .filter(
            (pl.col("Nucleus_Cell_Area") > 0.1) & (pl.col("Nucleus_Cell_Area") < 0.4)
        )
        .collect()
    )
    cell_IDs = cell_IDs.select("Metadata_CellID").to_series().to_list()

    # read in input data
    dframe = (
        pl.scan_parquet(input_path)
        .with_columns(
            pl.concat_str(
                [
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber",
                ],
                separator="_",
            ).alias("Metadata_CellID")
        )
        .filter(pl.col("Metadata_CellID").is_in(cell_IDs))
        .collect()
    )

    # filter wells with NA (allele mixture or empty)
    dframe = dframe.filter(~pl.col("Metadata_symbol").is_null())

    # Create new Metadata_node_type annotations
    dframe = dframe.drop([
        "Metadata_node_type",
        "Metadata_control_type",
    ]).with_columns(
        pl.when(pl.col("Metadata_symbol") == pl.col("Metadata_gene_allele"))
        .then(pl.lit("disease_wt"))
        .otherwise(pl.lit("allele"))
        .alias("Metadata_node_type")
    )

    # Add control annotations
    dframe = dframe.with_columns(
        pl.when(pl.col("Metadata_gene_allele").is_in(TC))
        .then(pl.lit("TC"))
        .otherwise(pl.col("Metadata_node_type"))
        .alias("Metadata_node_type")
    )

    dframe = dframe.with_columns(
        pl.when(pl.col("Metadata_gene_allele").is_in(PC))
        .then(pl.lit("PC"))
        .otherwise(pl.col("Metadata_node_type"))
        .alias("Metadata_node_type")
    )

    dframe = dframe.with_columns(
        pl.when(pl.col("Metadata_gene_allele").is_in(NC))
        .then(pl.lit("NC"))
        .otherwise(pl.col("Metadata_node_type"))
        .alias("Metadata_node_type")
    )

    dframe = dframe.with_columns(
        pl.when(pl.col("Metadata_gene_allele").is_in(cPC))
        .then(pl.lit("cPC"))
        .otherwise(pl.col("Metadata_node_type"))
        .alias("Metadata_node_type")
    )
    
    dframe.write_parquet(output_path, compression="gzip")
