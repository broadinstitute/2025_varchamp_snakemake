import preprocess
import os

batch = config["Metadata_Batch"]
plates = os.listdir(f"inputs/single_cell_profiles/{batch}/")

rule parquet_convert:
    input: 
        "inputs/single_cell_profiles/{batch}/{plate}/{plate}.sqlite"
    output:
        "outputs/single_cell_profiles/{batch}/{plate}_raw.parquet"
    threads: workflow.cores * 0.1
    benchmark:
        "outputs/benchmarks/{batch}/parquet_convert_{plate}.bwa.benchmark.txt"
    run:
        preprocess.convert_parquet(*input, *output, thread=threads)


rule annotate:
    input:
        "outputs/single_cell_profiles/{batch}/{plate}_raw.parquet"
    output:
        "outputs/single_cell_profiles/{batch}/{plate}_annotated.parquet"
    benchmark:
        "outputs/benchmarks/{batch}/annotate_{plate}.bwa.benchmark.txt"
    run:
        platemap = preprocess.get_platemap(f'inputs/metadata/platemaps/{wildcards.batch}/barcode_platemap.csv', f'{wildcards.plate}')
        platemap_path = f"inputs/metadata/platemaps/{wildcards.batch}/platemap/{platemap}.txt"
        preprocess.annotate_with_platemap(*input, platemap_path, *output)


rule aggregate:
    input:
        expand(
            "outputs/single_cell_profiles/{batch}/{plate}_annotated.parquet",
            batch=batch, 
            plate=plates)
    output:
        "outputs/batch_profiles/{batch}/profiles.parquet"
    benchmark:
        "outputs/benchmarks/{batch}/aggregate.bwa.benchmark.txt"
    run:
        preprocess.aggregate(input, *output)