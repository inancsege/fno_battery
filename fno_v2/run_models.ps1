# PowerShell script to run different models on the battery dataset
# Usage: .\run_models.ps1 [model_type] [dataset_type]

param (
    [string]$model_type = $null,
    [string]$dataset_type = $null
)

# Check if model_type is provided and valid
$valid_models = @("FNO", "LSTM", "LSTM_ATTN", "TCN", "XGBoost", "RandomForest", "LinearRegression", "SVR")
$valid_datasets = @("NASA_VIT", "NASA_RUL", "IEEE_FC", "XJTU", "GOLF_CAR")

if ($model_type -and -not ($valid_models -contains $model_type)) {
    Write-Host "Error: Invalid model type. Choose from: $($valid_models -join ', ')"
    exit 1
}

if ($dataset_type -and -not ($valid_datasets -contains $dataset_type)) {
    Write-Host "Error: Invalid dataset type. Choose from: $($valid_datasets -join ', ')"
    exit 1
}

# Build command with arguments
$cmd = "python main.py"
if ($model_type) {
    $cmd += " --model $model_type"
}
if ($dataset_type) {
    $cmd += " --dataset $dataset_type"
}

Write-Host "Running command: $cmd"
Write-Host "----------------------------------------------------"

# Execute the command
Invoke-Expression $cmd 