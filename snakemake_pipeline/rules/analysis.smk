import classification

rule calculate_metrics:
    input:
        "outputs/results/{batch}/{pipeline}/classifier_info.csv",
        "outputs/results/{batch}/{pipeline}/predictions.parquet"
    output:
        "outputs/analyses/{batch}/{pipeline}/metrics.csv"
    benchmark:
        "outputs/benchmarks/{batch}/{pipeline}_calc_metrics.bwa.benchmark.txt"
    run:
        classification.calculate_class_metrics(
            *input, 
            *output
        )


rule compute_hits:
    input:
        "outputs/analyses/{batch}/{pipeline}/metrics.csv"
    output:
        "outputs/analyses/{batch}/{pipeline}/metrics_summary.csv"
    benchmark:
        "outputs/benchmarks/{batch}/{pipeline}_comp_hits.bwa.benchmark.txt"
    run:
        classification.compute_hits(
            *input, 
            *output,
            trn_imbal_thres = config["trn_imbal_thres"],
            min_num_classifier = config["min_num_classifier"]
        )