import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import os
import argparse
import copy
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import optuna
import wandb

from dataset_position import SwinPrognosisDataset
from train_utils_new import train_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def objective(trial, args, train_val_indices, test_indices, external_indices, model_params, criterion_params):
    """Execute hyperparameter optimization with 10-fold cross-validation."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-5])
    dropout = trial.suggest_categorical('dropout', [0.5, 0.6, 0.7, 0.8])
    emb_dropout = trial.suggest_categorical('emb_dropout', [0.5, 0.6, 0.7, 0.8])
    depth = trial.suggest_categorical('depth', [2])
    heads = trial.suggest_categorical('heads', [2])
    dim = trial.suggest_categorical('dim', [256])
    mlp_dim = trial.suggest_categorical('mlp_dim', [256])
    dim_head = dim // heads
    alpha = trial.suggest_categorical('alpha', [0.5, 0.7])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-4, 1e-3, 1e-2])
    l1_lambda = trial.suggest_loguniform('l1_lambda', 1e-6, 1e-2)
    pool = trial.suggest_categorical('pool', ['cls', 'mean'])
    batch_size = trial.suggest_categorical('batch_size', [16])
    use_l1_loss = trial.suggest_categorical('use_l1_loss', [False])

    # Deep copy parameters to avoid modification of original
    model_params = copy.deepcopy(model_params)
    criterion_params = copy.deepcopy(criterion_params)

    # Update parameters
    model_params.update({
        'dim': dim,
        'depth': depth,
        'heads': heads,
        'mlp_dim': mlp_dim,
        'dim_head': dim_head,
        'dropout': dropout,
        'emb_dropout': emb_dropout,
        'pool': pool
    })
    criterion_params['alpha'] = alpha

    print(f"Trial {trial.number} - Updated model parameters: {model_params}")

    # Initialize Wandb for logging
    wandb.init(
        project='Survival_Analysis_Optimization',
        name=f"trial_{trial.number}",
        config={
            'learning_rate': learning_rate,
            'dropout': dropout,
            'emb_dropout': emb_dropout,
            'depth': depth,
            'heads': heads,
            'dim': dim,
            'mlp_dim': mlp_dim,
            'dim_head': dim_head,
            'alpha': alpha,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'l1_lambda': l1_lambda,
            'max_epochs': args.max_epochs,
            'seed': args.seed,
            'trial_number': trial.number,
        }
    )

    # 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    val_c_index_sum = 0

    # Prepare dataset and labels
    full_dataset = SwinPrognosisDataset(args.train_csv, pt_dir=args.data_dir)
    labels = [full_dataset[i][5] for i in train_val_indices]

    for fold, (train_indices, val_indices) in enumerate(skf.split(train_val_indices, labels)):
        print(f"Starting trial {trial.number}, fold {fold + 1}...")

        # Create fold directory
        trial_fold_dir = os.path.join(args.save_dir, f'trial_{trial.number}', f'fold_{fold+1}')
        os.makedirs(trial_fold_dir, exist_ok=True)

        # Train model and calculate validation C-index
        test_c_index, val_c_index, external_c_index = train_model(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            external_indices=external_indices,
            args=args,
            model_params=model_params,
            criterion_params=criterion_params,
            fold=fold + 1,
            trial_number=trial.number,
            learning_rate=learning_rate,
            use_l1_loss=use_l1_loss
        )

        val_c_index_sum += val_c_index

        # Save fold results
        fold_result = {
            'fold': fold + 1,
            'test_c_index': test_c_index,
            'val_c_index': val_c_index,
            'external_c_index': external_c_index,
        }
        fold_results_df = pd.DataFrame([fold_result])
        fold_results_file = os.path.join(trial_fold_dir, f'fold_{fold+1}_results.csv')
        fold_results_df.to_csv(fold_results_file, index=False)

    # Calculate average validation C-index
    avg_c_index = val_c_index_sum / 10

    # Save trial results to CSV
    trial_result = {
        'trial_number': trial.number,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'emb_dropout': emb_dropout,
        'depth': depth,
        'heads': heads,
        'dim': dim,
        'mlp_dim': mlp_dim,
        'dim_head': dim_head,
        'alpha': alpha,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'l1_lambda': l1_lambda,
        'avg_val_c_index': avg_c_index
    }

    results_file = os.path.join(args.save_dir, 'all_trials_results.csv')
    results_df = pd.DataFrame([trial_result])
    if not os.path.exists(results_file):
        results_df.to_csv(results_file, index=False)
    else:
        results_df.to_csv(results_file, mode='a', header=False, index=False)

    wandb.finish()
    return avg_c_index


def cross_validation_with_test_set(args, model_params, criterion_params):
    """Validate optimal hyperparameters through 10-fold cross-validation."""
    # Load dataset
    full_dataset = SwinPrognosisDataset(args.train_csv, pt_dir=args.data_dir)
    labels = [full_dataset[i][5] for i in range(len(full_dataset))]
    
    # Split train+val and test sets
    train_val_indices, test_indices, train_labels, test_labels = train_test_split(
        np.arange(len(full_dataset)), labels, test_size=0.1, stratify=labels, random_state=args.seed
    )
    
    # Load external test set
    external_dataset = SwinPrognosisDataset(args.external_csv, pt_dir=args.external_data_dir)
    external_indices = np.arange(len(external_dataset))

    # Create Optuna study
    study = optuna.create_study(direction='maximize')

    # Run hyperparameter optimization
    study.optimize(lambda trial: objective(
        trial,  
        args=args,
        train_val_indices=train_val_indices,
        test_indices=test_indices,
        external_indices=external_indices,
        model_params=model_params,
        criterion_params=criterion_params
    ), n_trials=30)

    # Output best hyperparameters and results
    print("Best hyperparameters: ", study.best_params)
    print("Best validation C-Index: ", study.best_value)


def main():
    parser = argparse.ArgumentParser(description="Training Transformer for Survival Analysis")
    parser.add_argument('--data_dir', type=str, default='/mnt/usb5/jijianxin/', 
                       help='Directory containing the data files')
    parser.add_argument('--train_csv', type=str, required=True, 
                       help='Path to the training CSV file')
    parser.add_argument('--external_csv', type=str, required=True, 
                       help='Path to the external test CSV file')
    parser.add_argument('--external_data_dir', type=str, required=True, 
                       help='Directory for the external dataset')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for DataLoader')
    parser.add_argument('--val_batch_size', type=int, default=8, 
                       help='Batch size for validation DataLoader')
    parser.add_argument('--test_batch_size', type=int, default=1, 
                       help='Batch size for test DataLoader')
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100, 
                       help='Max number of epochs')
    parser.add_argument('--save_dir', type=str, required=True, 
                       help='Directory to save the best model and results')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Weighting factor for the uncensored loss term')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--use_optuna', default=True, 
                       help='Use Optuna for hyperparameter tuning')
    parser.add_argument('--mode', type=str, choices=['train_val_test', 'cross_validation'], 
                       required=True, help='Choose between 7:1.5:1.5 split or cross-validation')

    args = parser.parse_args()
    set_seed(args.seed)

    # Default model and criterion parameters
    model_params = {
        'num_classes': 4,
        'input_dim': 1024,
        'dim': 512,
        'depth': 1,
        'heads': 2,
        'mlp_dim': 128,
        'pool': 'cls',
        'dim_head': 128,
        'dropout': 0.3,
        'emb_dropout': 0.3
    }
    criterion_params = {'alpha': args.alpha}

    if args.use_optuna:
        if args.mode == 'cross_validation':
            cross_validation_with_test_set(args, model_params, criterion_params)
        else:
            print("Currently only cross_validation mode is supported.")
    else:
        print("Starting default training without Optuna...")


if __name__ == "__main__":
    main()
