# Project Progress Tracker

## 20250907

I've implemented the new GFP filtering classification for the snakemake pipeline and I've run through the analysis for Batch 11 and 12.

The current issue is that for batch 7,8,13,14,15,16,18,19, the non-GFP filtering classifier INCLUDED the GFP_IntegratedIntensity feature as a feature in the classification process, so the AUROCs for standard classification may be inaccurate. We need to drop this column in the standard pipeline if the feature wasn't initially included after feat selection. But the morphological channels all work fine for now. To save time, we will now use the results for morphological results, and only re-run the GFP channel to speed up.