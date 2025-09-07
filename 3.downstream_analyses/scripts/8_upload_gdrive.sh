#!/bin/bash

# Base directories
IMG_BASE_DIR="../../2.snakemake_pipeline/outputs/visualize_imgs"
CELL_BASE_DIR="../../2.snakemake_pipeline/outputs/visualize_cells"
GDRIVE_BASE="gdrive_broad:IGVF/VarChAMP/VarCHAMP_WT_VAR_IMGs"

# Get all batch directories (excluding .ipynb_checkpoints)
BATCH_DIRS=$(ls -1 "$IMG_BASE_DIR" | grep -v "\.ipynb_checkpoints")

echo "Found batch directories:"
echo "$BATCH_DIRS"
echo ""

# Loop through each batch directory
for batch_id in $BATCH_DIRS; do
    echo "Processing batch: $batch_id"
    
    # Upload well images
    if [ -d "$IMG_BASE_DIR/$batch_id" ]; then
        echo "  Uploading well images for $batch_id..."
        rclone copy "$IMG_BASE_DIR/$batch_id/" "$GDRIVE_BASE/$batch_id/well_imgs/" \
            --include "*.png" --progress --transfers=16
    fi
    
    # Upload cell crop images
    if [ -d "$CELL_BASE_DIR/$batch_id" ]; then
        echo "  Uploading cell crop images for $batch_id..."
        rclone copy "$CELL_BASE_DIR/$batch_id/" "$GDRIVE_BASE/$batch_id/cell_imgs/" \
            --include "*.png" --progress --transfers=16
    fi
    
    echo "  Completed batch: $batch_id"
    echo ""
done

echo "All uploads completed!"