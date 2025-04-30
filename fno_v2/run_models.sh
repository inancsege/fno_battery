#!/bin/bash
# Bash script to run different models on the battery dataset
# Usage: ./run_models.sh [model_type] [dataset_type]

model_type=$1
dataset_type=$2

# Define valid options
valid_models=("FNO" "LSTM" "LSTM_ATTN" "TCN" "XGBoost" "RandomForest" "LinearRegression" "SVR")
valid_datasets=("NASA_VIT" "NASA_RUL" "IEEE_FC" "XJTU" "GOLF_CAR")

# Validate model type if provided
if [ ! -z "$model_type" ]; then
    valid_model=false
    for m in "${valid_models[@]}"; do
        if [ "$m" = "$model_type" ]; then
            valid_model=true
            break
        fi
    done
    
    if [ "$valid_model" = false ]; then
        echo "Error: Invalid model type. Choose from: ${valid_models[*]}"
        exit 1
    fi
fi

# Validate dataset type if provided
if [ ! -z "$dataset_type" ]; then
    valid_dataset=false
    for d in "${valid_datasets[@]}"; do
        if [ "$d" = "$dataset_type" ]; then
            valid_dataset=true
            break
        fi
    done
    
    if [ "$valid_dataset" = false ]; then
        echo "Error: Invalid dataset type. Choose from: ${valid_datasets[*]}"
        exit 1
    fi
fi

# Build command with arguments
cmd="python main.py"
if [ ! -z "$model_type" ]; then
    cmd="$cmd --model $model_type"
fi
if [ ! -z "$dataset_type" ]; then
    cmd="$cmd --dataset $dataset_type"
fi

echo "Running command: $cmd"
echo "----------------------------------------------------"

# Execute the command
eval $cmd 