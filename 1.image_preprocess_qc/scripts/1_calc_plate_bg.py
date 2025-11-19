import marimo

__generated_with = "0.8.22"
app = marimo.App(width="full")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Calculate plate background and conduct imaging well QC
        
        Calculate 384-well plate background and conduct the QC process for each imaging well
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(__file__):
    import argparse
    import os
    import re
    import glob
    import numpy as np
    import polars as pl
    from skimage.io import imread
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from img_utils import letter_dict, channel_dict

    # move into the script’s directory if needed
    # import pathlib
    # script_dir = pathlib.Path(__file__).resolve().parent
    # os.chdir(script_dir)

    letter_dict_rev = {v: k for k, v in letter_dict.items()}
    channel_dict_rev = {v: k for k, v in channel_dict.items()}
    PERCENTILE_ARRAY = np.array([25,50,75,80,90,95,99])

    # Check if running in marimo environment
    import sys
    running_in_marimo = 'marimo' in sys.modules and hasattr(sys, 'argv') and any('marimo' in arg for arg in sys.argv)

    if running_in_marimo:
        # Running in marimo - use default/hardcoded values
        batch_list = "2024_01_23_Batch_7,2024_02_06_Batch_8,2025_03_17_Batch_15,2025_03_17_Batch_16"
        # batch_list = "2024_12_09_Batch_11,2024_12_09_Batch_12"
        TIFF_IMG_DIR = "../inputs/cpg_imgs"
        output_dir = "../outputs/sum_stats_parquet"
        workers = 128
    else:
        # Running from command line - parse arguments
        p = argparse.ArgumentParser(__doc__)
        p.add_argument("--batch_list", help="batch to process")
        p.add_argument("--input_dir", type=str)
        p.add_argument("--output_dir", type=str)
        p.add_argument("--workers", type=int, default=256, help="Number of parallel workers")
        args = p.parse_args()

        batch_list = args.batch_list
        TIFF_IMG_DIR = args.input_dir
        output_dir = args.output_dir
        workers = args.workers

    batches = batch_list.split(",")
    return (
        PERCENTILE_ARRAY,
        TIFF_IMG_DIR,
        ThreadPoolExecutor,
        argparse,
        args,
        as_completed,
        batch_list,
        batches,
        channel_dict,
        channel_dict_rev,
        glob,
        imread,
        letter_dict,
        letter_dict_rev,
        np,
        os,
        output_dir,
        p,
        pl,
        re,
        sys,
        tqdm,
        workers,
    )


@app.cell
def __(mo):
    mo.md(r"## Calculate the plate background and summary statistics per plate and well")
    return


@app.cell
def __(
    PERCENTILE_ARRAY,
    channel_dict_rev,
    glob,
    imread,
    letter_dict_rev,
    np,
    re,
):
    def process_tiff_img(tiff_image_path, qs=PERCENTILE_ARRAY):
        """
        Process the tiff img and output its summary metrics
        """
        img = imread(tiff_image_path)

        ## get the metadata for the tiff img
        tiff_img_name = tiff_image_path.split("/")[-1]
        site = re.search(r"(?<=f)(\d{2})(?=p)", tiff_img_name.split('-')[0])[0]
        channel = channel_dict_rev[re.search(r"(?<=ch)(\d+)(?=sk)", tiff_img_name.split('-')[1])[0]]
        well_letter = letter_dict_rev[re.search(r'(?<=r)(\d{2})(?=c)', tiff_img_name.split('-')[0])[0]]
        well_num = re.search(r'(?<=c)(\d{2})(?=f)', tiff_img_name.split('-')[0])[0]
        well = f"{well_letter}{well_num}"
        img_metrics_dict = {
            "img_path": tiff_image_path, 
            "plate": tiff_image_path.split('/')[-3].split("__")[0],
            "well": well,
            "site": site, 
            "channel": channel
        }

        ## get the tiff img summary statistics
        if (len(img)>0):
            percentiles = dict(zip([f"perc_{q}" for q in qs], np.percentile(img, q=qs)))
        else:
            percentiles = dict(zip([f"perc_{q}" for q in qs], [np.nan]*len(qs)))

        img_metrics_dict.update(percentiles)
        return img_metrics_dict


    def summarize_tiff_img(path, max_gray=65535):
        """
        Summarize a single TIFF image (one site/channel of a well).
        Returns:
          n    -- number of pixels
          s    -- sum of pixel values (float64)
          ss   -- sum of squared pixel values (float64)
          hist -- histogram array shape (max_gray+1,)
        """
        img = imread(path)
        arr = img.ravel()
        n = arr.size

        # Pre-allocate histogram of zeros for full dynamic range
        hist = np.zeros(max_gray + 1, dtype=np.int64)
        if n:
            arr64 = arr.astype(np.float64, copy=False)
            # Sum and sum of squares
            s  = arr64.sum()
            ss = np.dot(arr64, arr64)
            # Only bincount up to actual max value to save a tiny bit of work
            maxi = int(arr.max())
            hist[: maxi + 1] = np.bincount(arr, minlength=maxi + 1)
        else:
            # Empty image → define sums as 0
            s = 0.0
            ss = 0.0
        return n, s, ss, hist


    def summarize_img(tiff_imgs, output_dict, percentiles=PERCENTILE_ARRAY, max_gray=65535):
        """
        Summarize TIFF images in sequence:
          - mean, std
          - arbitrary percentiles (given as fractions between 0 and 1)
        Modifies output_dict in place and also returns it.
        """
        if isinstance(tiff_imgs, str):
            tiff_imgs = glob.glob(tiff_imgs, recursive=True)

        if not tiff_imgs:
            # no images → fill with NaNs
            output_dict.update({f"perc_{int(p)}": None for p in percentiles})
            output_dict.update({"mean": None, "std": None})
            return output_dict

        # accumulators
        total_n = 0
        total_sum = 0.0
        total_sumsq = 0.0
        total_hist = np.zeros(max_gray + 1, dtype=np.int64)

        for path in tiff_imgs:
            n, s, ss, hist = summarize_tiff_img(path)
            total_n     += n
            total_sum   += s
            total_sumsq += ss
            total_hist  += hist

        mean = total_sum / total_n
        std  = np.sqrt(total_sumsq / total_n - mean * mean)

        # cumulative histogram for percentile lookup
        cum = np.cumsum(total_hist)
        pct_values = {
            f"perc_{int(p)}": int(np.searchsorted(cum, total_n * (p/100)))
            for p in percentiles
        }

        output_dict.update({"mean": mean, "std": std})
        output_dict.update(pct_values)
        return output_dict


    # def summarize_img_parallel(tiff_imgs, output_dict, workers=128, max_gray=65535):
    #     """
    #     Summarize TIFF images in parallel:
    #       - total pixel count (N)
    #       - total sum of intensities (S)
    #       - total sum of squares      (SS)
    #       - aggregated histogram over [0..max_gray]
    #       - derive mean, std, median

    #     Modifies output_dict in place and also returns it.
    #     """
    #     # expand glob‐style pattern if needed
    #     if isinstance(tiff_imgs, str):
    #         tiff_imgs = glob.glob(tiff_imgs, recursive=True)

    #     # prepare accumulators
    #     total_n = 0
    #     total_sum = 0.0
    #     total_sumsq = 0.0
    #     total_hist = np.zeros(max_gray + 1, dtype=np.int64)

    #     if not tiff_imgs:
    #         output_dict.update({"mean": None, "std": None, "median": None})
    #         return output_dict

    #     # parallel map + progress bar
    #     with ThreadPoolExecutor(max_workers=workers) as exe:
    #         it = exe.map(summarize_tiff_img, tiff_imgs)
    #         for n, s, ss, hist in tqdm(it, total=len(tiff_imgs), desc="Summarizing TIFFs"):
    #             total_n     += n
    #             total_sum   += s
    #             total_sumsq += ss
    #             total_hist  += hist

    #     # compute metrics
    #     mean = total_sum / total_n
    #     std  = np.sqrt(total_sumsq / total_n - mean * mean)

    #     # median via cumulative histogram
    #     cum = np.cumsum(total_hist)
    #     median = int(np.searchsorted(cum, total_n // 2))

    #     output_dict.update({"mean": mean, "std": std, "median": median})
    #     return output_dict
    return process_tiff_img, summarize_img, summarize_tiff_img


@app.cell
def __(
    TIFF_IMG_DIR,
    ThreadPoolExecutor,
    as_completed,
    batches,
    channel_dict_rev,
    glob,
    letter_dict_rev,
    os,
    output_dir,
    pl,
    re,
    summarize_img,
    tqdm,
    workers,
):
    for batch in batches:
        print(f"Summarize the per-channel plate-level summary statistics for {batch}:")
        plate_channel_imgs, plate_well_channel_imgs = [], []

        plates = os.listdir(f"{TIFF_IMG_DIR}/{batch}/images")
        for plate in tqdm(plates):
            for channel in channel_dict_rev.keys():
                ## map a tiff to its plate_well_channel
                plate_all_tiffs = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/*.tiff")
                plate_unique_wells = sorted(set([tiff.split('/')[-1][:6] for tiff in plate_all_tiffs]))
                for well in plate_unique_wells:
                    well_letter = letter_dict_rev[re.search(r'(?<=r)(\d{2})(?=c)', well)[0]]
                    well_num = re.search(r'(?<=c)(\d{2})', well)[0]
                    plate_well_channel = {
                        "plate": plate.split("__")[0], 
                        "well": f"{well_letter}{well_num}",
                        "channel": channel_dict_rev[channel]
                    }
                    channel_tiffs = f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/{well}*-ch{channel}sk*.tiff"
                    plate_well_channel_imgs.append((channel_tiffs, plate_well_channel))

                ## map a tiff to its plate_channel
                plate_channel_tiffs = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/*ch{channel}sk*.tiff", recursive=True)
                if plate_channel_tiffs:
                    plate_channel = {
                        "plate": plate.split("__")[0],
                        "channel": channel_dict_rev[channel]
                    }
                    plate_channel_imgs.append((plate_channel_tiffs, plate_channel))

        plate_sum_stats = []
        with ThreadPoolExecutor(max_workers=workers) as exe:
            # submit each (tiff_imgs, output_dict) pair as separate job
            ps_res = [
                exe.submit(summarize_img, tiff_imgs, output_dict)
                for tiff_imgs, output_dict in plate_channel_imgs
            ]
            for fut in tqdm(as_completed(ps_res), total=len(ps_res), desc="Processing tiffs per plate"):
                result = fut.result()
                plate_sum_stats.append(result)

        plate_well_sum_stats = []
        with ThreadPoolExecutor(max_workers=workers) as exe:
            # submit each (tiff_imgs, output_dict) pair as separate job
            pws_res = [
                exe.submit(summarize_img, tiff_imgs, output_dict)
                for tiff_imgs, output_dict in plate_well_channel_imgs
            ]
            for fut in tqdm(as_completed(pws_res), total=len(pws_res), desc="Processing tiffs per well"):
                result = fut.result()
                plate_well_sum_stats.append(result)

        if not os.path.exists(os.path.join(output_dir, batch)):
            os.makedirs(os.path.join(output_dir, batch))

        df_plate = pl.DataFrame(plate_sum_stats)
        df_plate.write_parquet(os.path.join(output_dir, batch, "plate_sum_stats.parquet"))

        df_plate_well = pl.DataFrame(plate_well_sum_stats, infer_schema_length=100000)
        df_plate_well.write_parquet(os.path.join(output_dir, batch, "plate_well_sum_stats.parquet"))
    return (
        batch,
        channel,
        channel_tiffs,
        df_plate,
        df_plate_well,
        exe,
        fut,
        plate,
        plate_all_tiffs,
        plate_channel,
        plate_channel_imgs,
        plate_channel_tiffs,
        plate_sum_stats,
        plate_unique_wells,
        plate_well_channel,
        plate_well_channel_imgs,
        plate_well_sum_stats,
        plates,
        ps_res,
        pws_res,
        result,
        well,
        well_letter,
        well_num,
    )


@app.cell
def __():
    # for batch in tqdm(batches):
    #     print(f"Summarize the per-channel well-level summary statistics for {batch}:")
    #     tiff_img_dict_mapper = []
    #     plates = os.listdir(f"{TIFF_IMG_DIR}/{batch}/images")
    #     for plate in tqdm(plates):
    #         all_tiffs = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/*.tiff")
    #         unique_wells = sorted(set([tiff.split('/')[-1][:6] for tiff in all_tiffs]))
    #         # print(len(unique_wells)) ## 384
    #         for well in tqdm(unique_wells):
    #             well_letter = letter_dict_rev[re.search(r'(?<=r)(\d{2})(?=c)', well)[0]]
    #             well_num = re.search(r'(?<=c)(\d{2})', well)[0]
    #             # print(well, result_dict["well"])
    #             for channel in channel_dict_rev.keys():
    #                 result_dict = {"plate": plate.split("__")[0], 
    #                                "well": f"{well_letter}{well_num}",
    #                                "channel": channel_dict_rev[channel]}
    #                 channel_tiffs = f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/{well}*-ch{channel}sk*.tiff" # glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/{well}*_ch{channel}sk*.tiff", recursive=True)[:100]
    #                 # print(channel_tiffs)
    #                 tiff_img_dict_mapper.append((channel_tiffs, result_dict))

    #     results_per_well = []
    #     with ThreadPoolExecutor(max_workers=workers) as exe:
    #         # submit each (tiff_imgs, output_dict) pair as separate job
    #         futures = [
    #             exe.submit(summarize_img_sequential, tiff_imgs, output_dict)
    #             for tiff_imgs, output_dict in tiff_img_dict_mapper
    #         ]
    #         for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing tiffs per well"):
    #             result = fut.result()
    #             results_per_well.append(result)

    #     df = pl.DataFrame(results_per_well, infer_schema_length=100000)
    #     df.write_parquet(os.path.join(output_dir, batch, "plate_well_sum_stats.parquet"))
    #     break


    # for batch in batches:
    #     print(f"Summarize the per-channel site info by plate for {batch}:")
    #     ## find all TIFFs
    #     paths = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/*/Images/*.tiff", recursive=True)
    #     records = []
    #     # process in parallel with a progress bar
    #     with ThreadPoolExecutor(max_workers=workers) as exe:
    #         futures = {exe.submit(process_tiff_img, p): p for p in paths}
    #         for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing TIFFs"):
    #             rec = fut.result()
    #             records.append(rec)

    #     if not os.path.exists(os.path.join(output_dir, batch)):
    #         os.makedirs(os.path.join(output_dir, batch))

    #     df = pl.DataFrame(records)
    #     df.write_parquet(os.path.join(output_dir, batch, "plate_site_channel.parquet"))
    return


if __name__ == "__main__":
    app.run()
