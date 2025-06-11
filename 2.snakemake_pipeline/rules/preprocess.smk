import preprocess
import utils

outputs = config["output_dir"]
# pipeline = config["pipeline"]

rule drop_empty_wells:
    input: 
        f"{outputs}/batch_profiles/{{batch}}/profiles.parquet",
    output: 
        f"{outputs}/batch_profiles/{{batch}}/profiles_tcdropped.parquet",
    benchmark:
        f"{outputs}/benchmarks/{{batch}}/profiles_tcdropped.bwa.benchmark.txt"
    run:
        preprocess.drop_empty_wells(
            *input, 
            *output, 
            pert_col=config["transfection_col"], 
            pert_name=config["trasfection_pert"]
        )


rule remove_nan:
    input:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}.parquet"
    output:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}_filtered.parquet"
    benchmark:
        f"{outputs}/benchmarks/{{batch}}/{{pipeline}}_filtered.bwa.benchmark.txt"
    params:
        drop_threshold = 100
    run:
        preprocess.drop_nan_features(
            *input, 
            *output, 
            cell_threshold=params.drop_threshold
        )


"""
## Well-position correction IS NOT applied for now, but it is always kept as a procedure to be tested
rule wellpos:
    input:
        f"{outputs}/batch_profiles/{{batch}}/filtered.parquet"
    output:
        f"{outputs}/batch_profiles/{{batch}}/filtered_wellpos.parquet"
    benchmark:
        f"{outputs}/benchmarks/wellpos_{{batch}}.bwa.benchmark.txt"
    params:
        parallel = config['parallel']
    run:
        preprocess.subtract_well_mean_polar(*input, *output)
"""


rule plate_stats:
    input:
        f"{outputs}/batch_profiles/{{batch}}/profiles_tcdropped_filtered.parquet"
    output:
        f"{outputs}/batch_profiles/{{batch}}/plate_stats.parquet"
    benchmark:
        f"{outputs}/benchmarks/plate_stats_{{batch}}.bwa.benchmark.txt"
    run:
        preprocess.compute_norm_stats_polar(*input, *output)


rule select_variant_feats:
    input:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}.parquet",
        f"{outputs}/batch_profiles/{{batch}}/plate_stats.parquet"
    output:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}_var.parquet",
    benchmark:
        f"{outputs}/benchmarks/{{batch}}/{{pipeline}}_var_{{batch}}.bwa.benchmark.txt"
    run:
        preprocess.select_variant_features_polars(*input, *output)


rule mad:
    input:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}.parquet",
        f"{outputs}/batch_profiles/{{batch}}/plate_stats.parquet"
    output:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}_mad.parquet"
    benchmark:
        f"{outputs}/benchmarks/{{batch}}/{{pipeline}}_mad_{{batch}}.bwa.benchmark.txt"
    run:
        preprocess.robustmad(input[0], input[1], *output)


rule outlier_removal:
    input: 
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}.parquet",
    output:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}_outlier.parquet",
    benchmark:
        f"{outputs}/benchmarks/{{batch}}/{{pipeline}}_outlier_{{batch}}.bwa.benchmark.txt"
    run:
        preprocess.clean.outlier_removal_polars(*input, *output)


rule feat_select:
    input:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}.parquet"
    output:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}_featselect.parquet"
    benchmark:
        f"{outputs}/benchmarks/{{batch}}/{{pipeline}}_feat_select_{{batch}}.bwa.benchmark.txt"
    run:
        preprocess.select_features(*input, *output)


rule filter_cells:
    input: 
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}.parquet"
    output:
        f"{outputs}/batch_profiles/{{batch}}/{{pipeline}}_filtcells.parquet"
    params:
        TC = config['TC'],
        NC = config['NC'],
        PC = config['PC'],
        cPC = config['cPC']
    run:
        preprocess.filter_cells(*input, *output, TC=params.TC, NC=params.NC, PC=params.PC, cPC=params.cPC)