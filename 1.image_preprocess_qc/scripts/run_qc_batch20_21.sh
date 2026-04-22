#!/bin/bash
# Run GPU-accelerated image QC for Batch 20 and 21
# Waits for downloads to complete first, then runs QC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Wait for downloads to complete
echo "Waiting for image downloads to complete..."

while pgrep -f "aws s3 sync.*2026_01_05_Batch_20" > /dev/null 2>&1; do
    B20_SIZE=$(du -sh /data/users/shenrunx/igvf/varchamp/2021_09_01_VarChAMP_imgs/2026_01_05_Batch_20/ 2>/dev/null | cut -f1)
    B21_SIZE=$(du -sh /data/users/shenrunx/igvf/varchamp/2021_09_01_VarChAMP_imgs/2026_01_05_Batch_21/ 2>/dev/null | cut -f1)
    echo "$(date): Batch 20: $B20_SIZE, Batch 21: $B21_SIZE - still downloading..."
    sleep 60
done

echo "$(date): Downloads complete!"

# Activate conda environment
source ~/software/anaconda3/etc/profile.d/conda.sh
conda activate varchamp

# Run GPU-accelerated QC with 2 GPUs (half of available 4)
echo "Running GPU-accelerated image QC..."
python scripts/calc_plate_bg_gpu.py \
    --batch_list "2026_01_05_Batch_20,2026_01_05_Batch_21" \
    --input_dir "inputs/cpg_imgs" \
    --platemaps_dir "../2.snakemake_pipeline/inputs/metadata/platemaps" \
    --output_dir "outputs/plate_bg_summary" \
    --gpus "0,1" \
    --batch_size 64 \
    --workers 32

echo "$(date): Image QC complete!"

# Run well QC flagging
echo "Running well QC flagging..."
python scripts/flag_qc_wells.py \
    --batch_ids "2026_01_05_Batch_20,2026_01_05_Batch_21" \
    --input_dir "outputs/plate_bg_summary" \
    --output_dir "outputs/plate_bg_summary" \
    2>&1 || echo "Note: flag_qc_wells.py not found, skipping QC flagging step"

echo "$(date): All done!"
