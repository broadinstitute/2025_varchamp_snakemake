# imports
import plotnine as plotnine
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm


# Define a function to compute metrics for each group
def compute_aubprc(auprc, prior):
    return (auprc * (1 - prior)) / ((auprc * (1 - prior)) + ((1 - auprc) * prior))


def compute_metrics(group):
    y_true = group["Label"].to_numpy()
    y_prob = group["Prediction"].to_numpy()
    y_pred = (y_prob > 0.5).astype(int)
    prior = sum(y_true == 1) / len(y_true)

    class_ID = group["Classifier_ID"].unique()[0]

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_prob)

    # Compute AUPRC
    auprc = average_precision_score(y_true, y_prob)
    aubprc = compute_aubprc(auprc, prior)

    # Compute macro-averaged F1 score
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Compute sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Compute balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "AUBPRC": aubprc,
        "Macro_F1": macro_f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced_Accuracy": balanced_acc,
        "Classifier_ID": class_ID,
    }


def calculate_class_metrics(classifier_info: str, predictions: str, metrics_file: str):
    batch_id = [subdir for subdir in classifier_info.split("/") if "Batch" in subdir][0]
    batch_id = f"B{batch_id.split('Batch_')[-1]}"

    # read in classifier info
    class_info = pl.read_csv(classifier_info)
    class_info = class_info.with_columns(
        (pl.col("trainsize_1") / (pl.col("trainsize_0") + pl.col("trainsize_1"))).alias(
            "train_prob_1"
        ),
        (pl.col("testsize_1") / (pl.col("testsize_0") + pl.col("testsize_1"))).alias(
            "test_prob_1"
        ),
    )

    # read in predictions
    preds = pl.scan_parquet(predictions)
    preds = preds.with_columns(pl.lit(batch_id).alias("Batch")).collect()
    preds = preds.with_columns(
        pl.concat_str(
            [pl.col("Classifier_ID"), pl.col("Metadata_Protein"), pl.col("Batch")],
            separator="_",
        ).alias("Full_Classifier_ID")
    )

    # Initialize an empty list to store the results
    results = []
    classIDs = preds.select("Full_Classifier_ID").to_series().unique().to_list()

    # Group by Classifier_ID and compute metrics for each group
    for id in tqdm(classIDs):
        metrics = compute_metrics(preds.filter(pl.col("Full_Classifier_ID") == id))
        metrics["Full_Classifier_ID"] = id
        results.append(metrics)

    # Convert the results to a Polars DataFrame
    metrics_df = pl.DataFrame(results)

    # Add classifier info and save
    metrics_df = metrics_df.join(class_info, on="Classifier_ID")
    metrics_df = metrics_df.with_columns(
        (
            pl.max_horizontal(["trainsize_0", "trainsize_1"])
            / pl.min_horizontal(["trainsize_0", "trainsize_1"])
        ).alias("Training_imbalance"),
        (
            pl.max_horizontal(["testsize_0", "testsize_1"])
            / pl.min_horizontal(["testsize_0", "testsize_1"])
        ).alias("Testing_imbalance"),
    )
    metrics_df.write_csv(metrics_file)

    return metrics_df


def compute_hits(metrics_file: str, metrics_summary_file: str, trn_imbal_thres: int, min_num_classifier: int):
    metrics_df = pl.read_csv(metrics_file)
    
    batch_id = [subdir for subdir in metrics_file.split("/") if "Batch" in subdir][0]
    batch_id = f"B{batch_id.split('Batch_')[-1]}"

    # Add useful columns (type, batch)
    metrics_df = metrics_df.with_columns(
        pl.format(
            "A{}P{}",  # 组合格式
            pl.col("Plate").str.extract(r'A(\d+)', 1),  # 提取A后的数字
            pl.col("Plate").str.extract(r'_P(\d+)T', 1)  # 提取P后的数字（排除T后的部分）
        ).alias("Allele_set")
    )

    metrics_df = metrics_df.with_columns(
        pl.when(pl.col("Full_Classifier_ID").str.contains("true"))
        .then(pl.lit("localization"))
        .otherwise(pl.lit("morphology"))
        .alias("Classifier_type"),
        pl.col("Full_Classifier_ID").str.split("_").list.last().alias("Batch"),
    )

    # Filter based on class imbalance
    metrics_ctrl = (
        metrics_df.filter(
            (pl.col("Training_imbalance") < trn_imbal_thres) & (pl.col("Metadata_Control"))
        )
        .select(["Classifier_type", "Batch", "AUROC"])
        .group_by(["Classifier_type", "Batch"])
        .quantile(0.99)
    ).rename({"AUROC": "AUROC_thresh"})

    # Merge with metrics_df and decide whether it passed the threshold
    metrics_df = metrics_df.join(metrics_ctrl, on=["Classifier_type", "Batch"])

    # Must be at least min_class_num classifiers per batch
    # Number of classifiers is the same for localization and morph, so just use morph
    classifier_count = (
        metrics_df.filter(
            (~pl.col("Metadata_Control"))
            & (pl.col("Classifier_type") == "localization")
        )
        .group_by(["allele_0", "Allele_set", "Batch", "allele_1"])
        .agg([pl.len().alias("Number_classifiers")])
    )
    classifier_count = classifier_count.pivot(
        index=["allele_0", "allele_1", "Allele_set"],
        on="Batch",
        values="Number_classifiers",
    )
    print("Total number of unique classifiers:", classifier_count.shape[0])
    print("Total number of unique variant alleles:", len(classifier_count.select("allele_0").to_series().unique().to_list()))
    print("Total number of unique WT genes:", len(classifier_count.select("allele_1").to_series().unique().to_list()))
    print("==========================================================================")

    # Must be at least min_class_num classifiers per batch
    # Number of classifiers is the same for localization and morph, so just use morph
    classifier_count = (
        metrics_df.filter(
            (pl.col("Training_imbalance") < trn_imbal_thres)
            & (~pl.col("Metadata_Control"))
            & (pl.col("Classifier_type") == "localization")
        )
        .group_by(["allele_0", "Allele_set", "Batch", "allele_1"])
        .agg([pl.len().alias("Number_classifiers")])
    )
    classifier_count = classifier_count.pivot(
        index=["allele_0", "allele_1", "Allele_set"],
        on="Batch",
        values="Number_classifiers",
    )
    print("After filtering out classifiers with training imbalance > 3:")
    print("Total number of unique classifiers:", classifier_count.shape)
    print("Total number of unique variant alleles:", len(classifier_count.select("allele_0").to_series().unique().to_list()))
    print("Total number of unique WT genes:", len(classifier_count.select("allele_1").to_series().unique().to_list()))
    print("==========================================================================")

    classifier_count = classifier_count.filter(
        (pl.col(batch_id) >= min_num_classifier)
    )
    print("After filtering out alleles with available number of classifiers < 2:")
    print("Total number of unique classifiers:", classifier_count.shape)
    print("Total number of unique variant alleles:", len(classifier_count.select("allele_0").to_series().unique().to_list()))
    print("Total number of unique WT genes:", len(classifier_count.select("allele_1").to_series().unique().to_list()))

    # filter based on this
    keep_alleles = classifier_count.select("allele_0").to_series().unique().to_list()
    metrics_df = metrics_df.filter(
        ~((~pl.col("Metadata_Control")) & ~pl.col("allele_0").is_in(keep_alleles))
    )

    # Filter by imbalance and calculate mean AUROC for each batch
    metrics_wtvar = (
        (
            metrics_df.filter(
                (pl.col("Training_imbalance") < trn_imbal_thres) & (~pl.col("Metadata_Control"))
            )
        )
        .select([
            "AUROC",
            "Classifier_type",
            "Batch",
            "AUROC_thresh",
            "allele_0",
            "trainsize_0",
            "testsize_0",
            "trainsize_1",
            "testsize_1",
            "Allele_set",
            "Training_imbalance",
        ])
        .group_by(["Classifier_type", "allele_0", "Allele_set", "Batch", "AUROC_thresh"])
        .agg([
            pl.all()
            .exclude(["Classifier_type", "allele_0", "Allele_set", "Batch", "AUROC_thresh"])
            .mean()
            .name.suffix("_mean")
        ])
    )
    
    # Write out results
    metrics_wtvar.write_csv(metrics_summary_file)
