import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir, title_suffix="", stage1_epochs=None):
    """
    Plot training and validation curves for loss and accuracy.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        train_accs (list): List of training accuracies
        val_accs (list): List of validation accuracies
        save_dir (str): Directory to save plots
        title_suffix (str): Suffix to add to plot titles
        stage1_epochs (int, optional): Number of epochs in stage 1 for two-stage training
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    # Add vertical line for stage separation if provided
    if stage1_epochs is not None:
        plt.axvline(x=stage1_epochs, color='g', linestyle='--', label='Stage 1 End')
    
    plt.title(f'Loss Curves{title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    
    # Add vertical line for stage separation if provided
    if stage1_epochs is not None:
        plt.axvline(x=stage1_epochs, color='g', linestyle='--', label='Stage 1 End')
    
    plt.title(f'Accuracy Curves{title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Save plot with proper naming
    clean_suffix = title_suffix.strip().lower().replace(' ', '_').replace('-', '_')
    if clean_suffix and not clean_suffix.startswith('_'):
        clean_suffix = '_' + clean_suffix
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'training_curves{clean_suffix}.png'))
    plt.close()

def plot_confusion_matrix(cm, save_dir, title="Confusion Matrix", model_name=None, rank=None, mode=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        save_dir (str): Directory to save plot
        title (str): Plot title
        model_name (str, optional): Model name for naming the file
        rank (int, optional): Rank of the model (for ranked models)
        mode (str, optional): Training mode (freeze, finetune, full)
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Create a descriptive filename
    if model_name and rank:
        # For ranked models
        filename = f'confusion_matrix_{model_name}_rank{rank}'
        if mode:
            filename += f'_{mode}'
        filename += '.png'
    elif model_name:
        # For best model
        filename = f'confusion_matrix_{model_name}'
        if mode:
            filename += f'_{mode}'
        filename += '.png'
    else:
        # Fall back to generic name with cleaned title
        clean_title = title.lower().replace(' ', '_').replace('-', '_')
        filename = f'{clean_title}.png'
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def save_metrics_to_csv(metrics, save_dir):
    """
    Save training metrics to CSV file.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        save_dir (str): Directory to save CSV
    """
    # Create a new dictionary with only simple list fields (avoid nested dicts)
    csv_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list) and not isinstance(value[0], dict) if value else True:
            csv_metrics[key] = value
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_metrics)
    df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)
    print(f"Metrics saved to {os.path.join(save_dir, 'training_metrics.csv')}")

def save_best_model_results(model, targets, predictions, num_classes, save_dir):
    """
    Save best model evaluation results.
    
    Args:
        model: PyTorch model
        targets (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Model predictions
        num_classes (int): Number of classes
        save_dir (str): Directory to save results
    """
    # Create confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate class-wise accuracy
    class_acc = np.diag(cm) / np.sum(cm, axis=1)
    
    # Save results to file
    with open(os.path.join(save_dir, 'best_model_results.txt'), 'w') as f:
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Overall accuracy: {np.mean(targets == predictions) * 100:.2f}%\n\n")
        f.write("Class-wise accuracy:\n")
        for i in range(num_classes):
            f.write(f"Class {i}: {class_acc[i] * 100:.2f}%\n")

def update_training_config(args, model, best_val_acc, total_time, save_dir):
    """
    Save training configuration and results.
    
    Args:
        args: Command line arguments
        model: PyTorch model
        best_val_acc (float): Best validation accuracy
        total_time (float): Total training time in seconds
        save_dir (str): Directory to save configuration
    """
    # Create config dictionary
    config = {
        'model': {
            'name': args.model_name,
            'pretrained': args.pretrained,
            'num_params': sum(p.numel() for p in model.parameters()),
            'num_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'training': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'optimizer': 'Adam',
            'initial_lr': args.learning_rate,
            'weight_decay': args.weight_decay,
            'augmentation': args.augment,
            'frame_mode': args.frame_mode,
            'num_frames': args.num_frames if hasattr(args, 'num_frames') else None
        },
        'results': {
            'best_val_acc': best_val_acc,
            'training_time_minutes': total_time / 60
        }
    }
    
    # Save config to YAML file
    with open(os.path.join(save_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nTraining configuration saved to: {os.path.join(save_dir, 'training_config.yaml')}")

def plot_advanced_metrics(metrics, save_dir, mode_suffix="", model_name=None):
    """
    Plot advanced metrics in one chart.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        save_dir (str): Directory to save plot
        mode_suffix (str): Mode suffix for file name
        model_name (str, optional): Model name for file naming
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    epochs = list(range(1, len(metrics['val_acc']) + 1))
    
    # Plot different metrics
    plt.plot(epochs, metrics['val_acc'], 'b-', label='Accuracy')
    
    # Check if F1 score is in metrics
    if 'val_f1' in metrics and metrics['val_f1']:
        plt.plot(epochs, metrics['val_f1'], 'r-', label='F1 Score')
    
    # Check if robust score is in metrics
    if 'val_robust_score' in metrics and metrics['val_robust_score']:
        plt.plot(epochs, metrics['val_robust_score'], 'g-', label='Robust Score')
    
    # Check if complete score is in metrics
    if 'val_complete_score' in metrics and metrics['val_complete_score']:
        plt.plot(epochs, metrics['val_complete_score'], 'y-', label='Complete Score')
    
    # Check if balanced score is in metrics
    if 'val_balanced_score' in metrics and metrics['val_balanced_score']:
        plt.plot(epochs, metrics['val_balanced_score'], 'm-', label='Balanced Score')
    
    plt.title('Advanced Metrics' + (' - ' + mode_suffix.title() if mode_suffix else ''))
    plt.xlabel('Epochs')
    plt.ylabel('Score (%)')
    plt.legend()
    plt.grid(True)
    
    # Save plot with proper naming
    clean_suffix = mode_suffix.lower().replace(' ', '_').replace('-', '_')
    if clean_suffix and not clean_suffix.startswith('_'):
        clean_suffix = '_' + clean_suffix
    
    filename = 'advanced_metrics'
    if model_name:
        filename += f'_{model_name}'
    filename += f'{clean_suffix}.png'
    
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def save_model_results(model_results, save_dir, model_name=None, rank=None, mode=None, selection_metric=None):
    """
    Save model results to plots directory.
    
    Args:
        model_results (dict): Dictionary with model results
        save_dir (str): Base directory for saving results
        model_name (str, optional): Name of model architecture
        rank (int, optional): Rank of model in top N models
        mode (str, optional): Training mode (freeze, finetune, full)
        selection_metric (str, optional): Metric used for model selection
    """
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot confusion matrix with descriptive name
    if 'confusion_matrix' in model_results:
        title_components = []
        if model_name:
            title_components.append(model_name)
        if mode:
            title_components.append(mode.title())
        if rank:
            title_components.append(f"Rank {rank}")
        
        title = "Confusion Matrix"
        if title_components:
            title += " - " + " ".join(title_components)
        
        plot_confusion_matrix(
            model_results['confusion_matrix'],
            save_dir,
            title=title,
            model_name=model_name,
            rank=rank,
            mode=mode
        )

def organize_config_files(save_dir, initial_config, training_results, mode=None):
    """
    Organize configuration files into a config directory.
    
    Args:
        save_dir (str): Base directory for saving
        initial_config (dict or str): Initial configuration (dict or path to yaml)
        training_results (dict or str): Training results (dict or path to yaml)
        mode (str, optional): Training mode for naming files
    """
    # Create config directory
    config_dir = os.path.join(save_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    # Handle initial config
    if isinstance(initial_config, str) and os.path.isfile(initial_config):
        # Load from file if it's a path
        with open(initial_config, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        # Use as is if it's already a dict
        config_data = initial_config
    
    # Save initial config
    config_filename = 'initial_config'
    if mode:
        config_filename += f'_{mode}'
    config_filename += '.yaml'
    
    with open(os.path.join(config_dir, config_filename), 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # Handle training results config
    if isinstance(training_results, str) and os.path.isfile(training_results):
        # Load from file if it's a path
        with open(training_results, 'r') as f:
            results_data = yaml.safe_load(f)
    else:
        # Use as is if it's already a dict
        results_data = training_results
    
    # Save training results
    results_filename = 'training_results'
    if mode:
        results_filename += f'_{mode}'
    results_filename += '.yaml'
    
    with open(os.path.join(config_dir, results_filename), 'w') as f:
        yaml.dump(results_data, f, default_flow_style=False)
    
    return config_dir

def create_unified_plots_structure(save_dir):
    """
    Create unified plots directory structure.
    
    Args:
        save_dir (str): Base directory for saving
        
    Returns:
        str: Path to plots directory
    """
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir 