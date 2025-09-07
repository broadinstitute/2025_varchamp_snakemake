import classification

outputs = config["output_dir"]
# pipeline = config["pipeline"]
batch = config["Metadata_Batch"]

rule classify:
    input:
        f"{outputs}/batch_profiles/{batch}/{{pipeline}}.parquet",
        f"{outputs}/batch_profiles/{batch}/profiles_tcdropped_filtered_var_mad_outlier.parquet",
    output:
        f"{outputs}/classification_results/{batch}/{{pipeline}}/feat_importance.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/classifier_info.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/predictions.parquet",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/feat_importance_gfp_adj.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/classifier_info_gfp_adj.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/predictions_gfp_adj.parquet",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/gfp_adj_filtered_cells_profiles.parquet"
    benchmark:
        f"{outputs}/benchmarks/{{pipeline}}_classify_{batch}.bwa.benchmark.txt"
    params:
        cc_thresh = config["cc_threshold"],
        plate_layout = config["plate_layout"],
    run:
        classification.run_classify_workflow(*input, *output, cc_threshold=params.cc_thresh, plate_layout=params.plate_layout)


rule calculate_metrics:
    input:
        f"{outputs}/classification_results/{batch}/{{pipeline}}/classifier_info.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/predictions.parquet"
    output:
        f"{outputs}/classification_analyses/{batch}/{{pipeline}}/metrics.csv"
    benchmark:
        f"{outputs}/benchmarks/{batch}/{{pipeline}}_calc_metrics.bwa.benchmark.txt"
    run:
        classification.calculate_class_metrics(
            *input, 
            *output
        )


rule calculate_metrics_gfp_adj:
    input:
        f"{outputs}/classification_results/{batch}/{{pipeline}}/classifier_info_gfp_adj.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/predictions_gfp_adj.parquet"
    output:
        f"{outputs}/classification_analyses/{batch}/{{pipeline}}/metrics_gfp_adj.csv"
    benchmark:
        f"{outputs}/benchmarks/{batch}/{{pipeline}}_calc_metrics.bwa.benchmark_gfp_adj.txt"
    run:
        classification.calculate_class_metrics(
            *input, 
            *output
        )


"""
rule compute_hits:
    input:
        f"{outputs}/classification_analyses/{batch}/{{pipeline}}/metrics.csv"
    output:
        f"{outputs}/classification_analyses/{batch}/{{pipeline}}/metrics_summary.csv"
    benchmark:
        f"{outputs}/benchmarks/{batch}/{{pipeline}}_comp_hits.bwa.benchmark.txt"
    run:
        classification.compute_hits(
            *input, 
            *output,
            trn_imbal_thres = config["trn_imbal_thres"],
            min_num_classifier = config["min_num_classifier"]
        )
"""