import preprocess
import os

inputs = config["input_dir"]
outputs = config["output_dir"]
batch = config["Metadata_Batch"]
plates = os.listdir(f"{inputs}/single_cell_profiles/{batch}/")

rule parquet_convert:
    input:
        f"{inputs}/single_cell_profiles/{batch}/{{plate}}/{{plate}}.sqlite"
    output:
        f"{outputs}/single_cell_profiles/{batch}/{{plate}}_raw.parquet"
    threads: workflow.cores * 0.1
    benchmark:
        f"{outputs}/benchmarks/{batch}/parquet_convert_{{plate}}.bwa.benchmark.txt"
    run:
        preprocess.convert_parquet(*input, *output, thread=threads)


rule annotate:
    input:
        f"{outputs}/single_cell_profiles/{batch}/{{plate}}_raw.parquet"
    output:
        f"{outputs}/single_cell_profiles/{batch}/{{plate}}_annotated.parquet"
    benchmark:
        f"{outputs}/benchmarks/{batch}/annotate_{{plate}}.bwa.benchmark.txt"
    run:
        platemap = preprocess.get_platemap(f'{inputs}/metadata/platemaps/{batch}/barcode_platemap.csv', f'{wildcards.plate}')
        platemap_path = f"{inputs}/metadata/platemaps/{batch}/platemap/{platemap}.txt"
        preprocess.annotate_with_platemap(*input, platemap_path, *output)


rule aggregate:
    input:
        expand(
            f"{outputs}/single_cell_profiles/{batch}/{{plate}}_annotated.parquet",
            batch=batch, 
            plate=plates)
    output:
        f"{outputs}/batch_profiles/{batch}/profiles.parquet"
    benchmark:
        f"{outputs}/benchmarks/{batch}/aggregate.bwa.benchmark.txt"
    run:
        preprocess.aggregate(input, *output)