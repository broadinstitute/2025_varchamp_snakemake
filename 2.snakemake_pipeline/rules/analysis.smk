import classification

outputs = config["output_dir"]
# pipeline = config["pipeline"]
batch = config["Metadata_Batch"]

rule classify:
    input:
        f"{outputs}/batch_profiles/{batch}/{{pipeline}}.parquet",
    output:
        f"{outputs}/classification_results/{batch}/{{pipeline}}/feat_importance.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/classifier_info.csv",
        f"{outputs}/classification_results/{batch}/{{pipeline}}/predictions.parquet"
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