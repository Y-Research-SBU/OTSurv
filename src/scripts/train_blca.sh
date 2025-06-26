#!/bin/bash

DEFAULT_CANCER_TYPE="BLCA"
DEFAULT_DEVICE="cuda:0"

CANCER_TYPE=${1:-$DEFAULT_CANCER_TYPE}
DEVICE=${2:-$DEFAULT_DEVICE} 

# Parameter ranges (case-sensitive values for variables)
K_VALUES=(0 1 2 3 4)

# Fixed parameter definitions
BASE_DATA_SOURCE="../data"
RESULTS_DIR="../result/exp_otsurv_train"
SPLIT_NAMES="train,val,test"
TARGET_COL="dss_survival_days"
IN_DIM="1024"
OPT="adamW"
LR="0.0001"
LR_SCHEDULER="cosine"
ACCUM_STEPS="1"
WD="0.00001"
WARMUP_EPOCHS="1"
MAX_EPOCHS="50"
TRAIN_BAG_SIZE="-1"
BATCH_SIZE="16"
SEED="1"
NUM_WORKERS="8"
LOSS_FN="cox"
NLL_ALPHA="0.5"                 
N_LABEL_BINS="4"                
EARLY_STOPPING="1"              
MODEL_TYPE="otsurv"
ES_METRIC="loss"

# Loop through all combinations
for K in "${K_VALUES[@]}"; do
  # Convert CANCER_TYPE to lowercase for path usage
  CANCER_TYPE_LOWER=$(echo "$CANCER_TYPE" | tr '[:upper:]' '[:lower:]')

  # Dynamically set DATA_SOURCE and SPLIT_DIR
  DATA_SOURCE="${BASE_DATA_SOURCE}/tcga_${CANCER_TYPE_LOWER}/extracted_mag20x_patch256_fp/extracted-vit_large_patch16_224.dinov2.uni_mass100k/feats_h5"
  SPLIT_DIR="survival/TCGA_${CANCER_TYPE}_overall_survival_k=${K}"
  
  echo "Running experiment: Cancer=${CANCER_TYPE}, k=${K}"
  echo "Using data source: ${DATA_SOURCE}"

  # Execute the Python script
  python -m training.main_survival \
    --data_source "$DATA_SOURCE" \
    --results_dir "$RESULTS_DIR" \
    --split_dir "$SPLIT_DIR" \
    --split_names "$SPLIT_NAMES" \
    --task "${CANCER_TYPE}_survival" \
    --target_col "$TARGET_COL" \
    --in_dim "$IN_DIM" \
    --opt "$OPT" \
    --lr "$LR" \
    --lr_scheduler "$LR_SCHEDULER" \
    --accum_steps "$ACCUM_STEPS" \
    --wd "$WD" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --max_epochs "$MAX_EPOCHS" \
    --train_bag_size "$TRAIN_BAG_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --loss_fn "$LOSS_FN" \
    --nll_alpha "$NLL_ALPHA" \
    --n_label_bins "$N_LABEL_BINS" \
    --early_stopping "$EARLY_STOPPING" \
    --model_type "$MODEL_TYPE" \
    --es_metric "$ES_METRIC" \
    --device "$DEVICE"

  echo "Experiment completed: Cancer=${CANCER_TYPE}, k=${K}"
done

