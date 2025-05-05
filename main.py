# -*- coding: utf-8 -*-
"""
Main script for the FNO Battery Capacity Prediction project
"""

import os
import sys
import argparse
import torch
import numpy as np
import logging
from datetime import datetime

from utils.logger import setup_logger
from config import MODEL_CONFIGS, DATASET_CONFIGS, GENERAL_CONFIG
import models
from preprocessing.data_processor import DataProcessor
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Ensure all paths are absolute
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    results_dir = os.path.join(outputs_dir, "results")
    figures_dir = os.path.join(outputs_dir, "figures")
    logs_dir = os.path.join(base_dir, "logs")
    
    # Create directories with absolute paths
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Battery Capacity Prediction with Various Models')
    
    parser.add_argument('--model', type=str, required=False, default='FNO',
                        choices=['FNO', 'LSTM', 'LSTM_ATTN', 'TCN', 'XGBoost', 
                                'RandomForest', 'LinearRegression', 'SVR'],
                        help='Type of model to use')
    
    parser.add_argument('--dataset', type=str, required=False, default='NASA_VIT',
                        choices=['NASA_VIT', 'NASA_RUL', 'IEEE_FC', 'XJTU', 'GOLF_CAR'],
                        help='Dataset to use for training and evaluation')
    
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length for time series data (overrides config)')
    
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation on test data')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose logging')
    
    return parser.parse_args()

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Main function to run the training and evaluation pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up directories
    setup_directories()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{args.model}_{args.dataset}_{timestamp}.log"
    logger = setup_logger(log_file, log_level)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Select device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    
    # Get model and dataset configurations
    model_config = MODEL_CONFIGS[args.model].copy()
    dataset_config = DATASET_CONFIGS[args.dataset].copy()
    
    # Override configurations with command line arguments if provided
    if args.seq_len is not None:
        model_config['seq_len'] = args.seq_len
    if args.batch_size is not None:
        model_config['batch_size'] = args.batch_size
    if args.epochs is not None:
        model_config['epochs'] = args.epochs
    if args.learning_rate is not None:
        model_config['learning_rate'] = args.learning_rate
    
    # Log configurations
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model Config: {model_config}")
    logger.info(f"Dataset Config: {dataset_config}")
    
    # Process data
    logger.info("Processing data...")
    data_processor = DataProcessor(
        dataset_name=args.dataset,
        **dataset_config,
        **{k: v for k, v in model_config.items() if k in ['seq_len', 'batch_size']}
    )
    
    train_loader, val_loader, test_loader, dataset_info = data_processor.get_data_loaders()
    logger.info(f"Dataset info: {dataset_info}")
    
    # Initialize model
    logger.info("Initializing model...")
    ModelClass = getattr(models, args.model)
    
    # Handle model initialization based on model type
    if args.model in ['LSTM', 'LSTM_ATTN', 'TCN']:
        # These models don't use seq_len in initialization
        model = ModelClass(
            input_dim=dataset_info['input_dim'],
            output_dim=dataset_info['output_dim'],
            **{k: v for k, v in model_config.items() if k not in ['seq_len', 'batch_size', 'epochs', 'learning_rate']}
        )
    elif args.model in ['FNO', 'FNO_RUL']:
        # FNO models use seq_len
        model = ModelClass(
            input_dim=dataset_info['input_dim'],
            output_dim=dataset_info['output_dim'],
            seq_len=model_config['seq_len'],
            **{k: v for k, v in model_config.items() if k not in ['seq_len', 'batch_size', 'epochs', 'learning_rate']}
        )
    elif args.model == 'RandomForest':
        # RandomForest only accepts specific parameters
        model = ModelClass(
            seq_len=model_config['seq_len'],
            input_dim=dataset_info['input_dim'],
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 10)
        )
    elif args.model == 'XGBoost':
        # XGBoost with its specific parameters
        model = ModelClass(
            seq_len=model_config['seq_len'],
            input_dim=dataset_info['input_dim'],
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 5),
            learning_rate=model_config.get('learning_rate', 0.1),
            use_cuda=model_config.get('use_cuda', False)
        )
    elif args.model == 'LinearRegression':
        # LinearRegression with minimal parameters
        model = ModelClass(
            seq_len=model_config['seq_len'],
            input_dim=dataset_info['input_dim']
        )
    elif args.model == 'SVR':
        # SVR with its specific parameters
        model = ModelClass(
            seq_len=model_config['seq_len'],
            input_dim=dataset_info['input_dim'],
            kernel=model_config.get('kernel', 'rbf'),
            C=model_config.get('C', 1.0),
            epsilon=model_config.get('epsilon', 0.1)
        )
    else:
        # Default initialization for other models
        model = ModelClass(
            input_dim=dataset_info['input_dim'],
            output_dim=dataset_info['output_dim'],
            seq_len=model_config['seq_len'],
            **{k: v for k, v in model_config.items() if k not in ['seq_len', 'batch_size', 'epochs', 'learning_rate']}
        )
    
    # Move model to device
    model = model.to(device)
    
    # Model save path
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_filename = f"{args.model}_{args.dataset}_model.pt"
    model_save_path = os.path.join(model_save_dir, model_filename)
    
    # Results and figure paths
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "results")
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
    results_file = os.path.join(results_dir, f"{args.model}_{args.dataset}_results.json")
    figure_path = os.path.join(figures_dir, f"{args.model}_{args.dataset}_prediction.png")
    
    if not args.eval_only:
        # Train model
        logger.info("Training model...")
        
        # Prepare trainer config
        trainer_config = {
            'epochs': model_config['epochs'],
            'learning_rate': model_config['learning_rate'],
            'model_save_path': model_save_path,
            'device': device,
            'logger': logger,
            'model_type': args.model
        }
        
        # Custom configuration for FNO model
        if args.model == 'FNO':
            # Special optimizer and scheduler for FNO
            trainer_config['optimizer_type'] = 'AdamW'
            trainer_config['weight_decay'] = 0.01
            trainer_config['lr_scheduler'] = 'cosine_warmup'
            trainer_config['warmup_epochs'] = 10
            trainer_config['min_lr'] = 1e-6
            
            # Gradient clipping to stabilize training
            trainer_config['grad_clip_value'] = 1.0
            
            # Use aggressive EMA for validation with moving average
            trainer_config['use_ema'] = True
            trainer_config['ema_decay'] = 0.998
            
            # Enhanced loss function with frequency-domain penalties
            trainer_config['use_spectral_loss'] = True
            trainer_config['spectral_loss_weight'] = 0.02
            
            # Train for more epochs with patience-based stopping
            trainer_config['patience'] = 20
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config
        )
        trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model...")
    
    # Skip evaluation if test set is too small
    test_set_size = len(test_loader.dataset)
    if test_set_size < 2:
        logger.warning(f"Test set too small ({test_set_size} samples). Skipping evaluation.")
        logger.info("Done!")
        return
    
    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        config={
            'device': device,
            'model_save_path': model_save_path,
            'results_file': results_file,
            'figure_path': figure_path,
            'dataset_info': dataset_info,
            'logger': logger
        }
    )
    evaluation_result = evaluator.evaluate()
    
    # Handle different return formats from evaluator
    if isinstance(evaluation_result, tuple) and len(evaluation_result) == 3:
        # Returns: all_targets, all_preds, metrics
        _, _, metrics = evaluation_result
    else:
        metrics = evaluation_result
    
    # Print evaluation results
    logger.info("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Prediction plot saved to {figure_path}")
    logger.info("Done!")

if __name__ == "__main__":
    main() 