'''Download and prepare data.'''
from cytotable import convert
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import random

COLUMNS = (
    "TableNumber",
    "ImageNumber",
    "ObjectNumber",
    "Metadata_Well",
    "Metadata_Plate",
    "Parent_Cells",
    "Parent_Nuclei",
    "Cytoplasm_Parent_Cells",
    "Cytoplasm_Parent_Nuclei",
)

COMMANDS =  """
            WITH Image_Filtered AS (
                SELECT
                    Metadata_TableNumber,
                    Metadata_ImageNumber,
                    Metadata_Well,
                    Metadata_Plate
                FROM
                    read_parquet('image.parquet')
                )
            SELECT
                *
            FROM
                Image_Filtered AS image
            JOIN read_parquet('cytoplasm.parquet') AS cytoplasm ON
                cytoplasm.Metadata_TableNumber = image.Metadata_TableNumber
                AND cytoplasm.Metadata_ImageNumber = image.Metadata_ImageNumber
            JOIN read_parquet('cells.parquet') AS cells ON
                cells.Metadata_TableNumber = cytoplasm.Metadata_TableNumber
                AND cells.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
                AND cells.Metadata_ObjectNumber = cytoplasm.Metadata_Cytoplasm_Parent_Cells
            JOIN read_parquet('nuclei.parquet') AS nuclei ON
                nuclei.Metadata_TableNumber = cytoplasm.Metadata_TableNumber
                AND nuclei.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
                AND nuclei.Metadata_ObjectNumber = cytoplasm.Metadata_Cytoplasm_Parent_Nuclei
            """



def convert_parquet(
    input_file,
    output_file,
    cols=COLUMNS,
    chunk_size=150000,
    joins=COMMANDS,
    thread=2,
):
    """Convert sqlite profiles to parquet"""

    hash_str = str(random.getrandbits(128))
    parsl_config = Config(
                        executors=[
                            ThreadPoolExecutor(
                                max_threads=thread
                            )
                        ],
                        run_dir=f'./runinfo/{hash_str}'
                    )
   
    convert(
        source_path=input_file,
        dest_path=output_file,
        identifying_columns=cols,
        dest_datatype='parquet',
        chunk_size=chunk_size,
        preset="cell-health-cellprofiler-to-cytominer-database",
        joins=joins,
        reload_parsl_config=True,
        parsl_config=parsl_config
    )
