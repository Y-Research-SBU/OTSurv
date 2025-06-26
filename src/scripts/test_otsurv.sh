#!/bin/bash

DEFAULT_DEVICE="cuda:0"
DEVICE=${1:-$DEFAULT_DEVICE} 

# Parameter ranges (case-sensitive values for variables)
K_VALUES=(0 1 2 3 4)
CANCER_TYPES=("BLCA" "BRCA" "COADREAD" "KIRC" "LUAD" "STAD")  # Uppercase for task names

# Fixed parameter definitions
BASE_DATA_SOURCE="../data"
RESULTS_DIR="../result/exp_otsurv_test"
SPLIT_NAMES="test"
TARGET_COL="dss_survival_days"
IN_DIM="1024"
BATCH_SIZE="16"
SEED="1"
NUM_WORKERS="8"
LOSS_FN="cox"
NLL_ALPHA="0.5"                 
N_LABEL_BINS="4"                             
MODEL_TYPE="otsurv"

# Loop through all combinations
for CANCER_TYPE in "${CANCER_TYPES[@]}"; do
  for K in "${K_VALUES[@]}"; do
    # Convert CANCER_TYPE to lowercase for path usage
    CANCER_TYPE_LOWER=$(echo "$CANCER_TYPE" | tr '[:upper:]' '[:lower:]')

    # Dynamically set DATA_SOURCE and SPLIT_DIR
    DATA_SOURCE="${BASE_DATA_SOURCE}/tcga_${CANCER_TYPE_LOWER}/extracted_mag20x_patch256_fp/extracted-vit_large_patch16_224.dinov2.uni_mass100k/feats_h5"
    SPLIT_DIR="survival/TCGA_${CANCER_TYPE}_overall_survival_k=${K}"
    
    echo "Running experiment: Cancer=${CANCER_TYPE}, k=${K}"
    echo "Using data source: ${DATA_SOURCE}"

    CHECKPOINT_PATH="../checkpoints/model_${CANCER_TYPE_LOWER}_fold${K}.pth"

    # Execute the Python script
    python -m training.test_survival \
    --checkpoint_path "$CHECKPOINT_PATH" \
      --data_source "$DATA_SOURCE" \
      --results_dir "$RESULTS_DIR" \
      --split_dir "$SPLIT_DIR" \
      --split_names "$SPLIT_NAMES" \
      --task "${CANCER_TYPE}_survival" \
      --target_col "$TARGET_COL" \
      --in_dim "$IN_DIM" \
      --batch_size "$BATCH_SIZE" \
      --seed "$SEED" \
      --num_workers "$NUM_WORKERS" \
      --loss_fn "$LOSS_FN" \
      --nll_alpha "$NLL_ALPHA" \
      --n_label_bins "$N_LABEL_BINS" \
      --model_type "$MODEL_TYPE" \
      --device "$DEVICE"

    echo "Experiment completed: Cancer=${CANCER_TYPE}, k=${K}"
  done
done
