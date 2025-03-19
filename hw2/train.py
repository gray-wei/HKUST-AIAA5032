import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Import custom modules
from utils.data_utils import create_data_loaders
from utils.training_utils import (
    print_system_info, set_seed, train_one_epoch, 
    validate, print_training_info
)
from utils.visualization import (
    plot_training_curves, save_metrics_to_csv,
    plot_confusion_matrix, plot_advanced_metrics,
    save_model_results, organize_config_files,
    create_unified_plots_structure
)
from models.cnn_model import VideoClassifier

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train video classifier")
    
    # Training mode selection
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['freeze', 'finetune', 'full'],
                        help='Training mode: freeze (backbone frozen), finetune (full network), full (from scratch)')
    
    # Data related arguments
    parser.add_argument('--data_dir', type=str, default='video_frames_30fpv_320p', 
                        help='Directory with video frames')
    parser.add_argument('--csv_file', type=str, default='data/trainval.csv', 
                        help='CSV file with video IDs and labels')
    parser.add_argument('--val_split', type=float, default=0.2, 
                        help='Proportion of data to use for validation')
    parser.add_argument('--frame_mode', type=str, default='uniform', 
                        choices=['middle', 'random', 'all', 'uniform', 'key_frames'], 
                        help='How to select frames from videos')
    parser.add_argument('--num_frames', type=int, default=6,
                        help='Number of frames to select when mode is uniform (3-30)')
    
    # Model related arguments
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', 
                        help='Name of the pre-trained model to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pre-trained weights')
    parser.add_argument('--temporal_module', type=str, default='attention',
                        choices=['transformer', 'lstm', 'attention'],
                        help='Temporal modeling module type')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate for classifier')
    parser.add_argument('--temporal_dropout', type=float, default=0.2,
                        help='Dropout rate for temporal module')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha parameter (0 to disable)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing parameter')
    
    # Training related arguments
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate for training')
    parser.add_argument('--backbone_lr_factor', type=float, default=0.1,
                        help='Factor to multiply backbone learning rate (finetune mode)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay for regularization')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Reduction factor for ReduceLROnPlateau scheduler')
    
    # Save related arguments
    parser.add_argument('--save_dir', type=str, default='results', 
                        help='Directory to save models and results')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Epoch frequency for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--keep_best_models', type=int, default=3,
                        help='Number of best models to keep')
    
    # Model selection metric
    parser.add_argument('--model_selection', type=str, default='robust_score',
                        choices=['accuracy', 'macro_f1', 'balanced_score', 'robust_score', 'complete_score', 'balanced_acc'],
                        help='Metric for selecting best model')
    
    args = parser.parse_args()
    
    # Validate num_frames based on frame_mode
    if args.frame_mode == 'uniform':
        if args.num_frames < 3 or args.num_frames > 30:
            parser.error("For 'uniform' mode, num_frames must be between 3 and 30")
    elif args.num_frames != 6:
        print(f"\nWarning: --num_frames is only used in 'uniform' mode. "
              f"In '{args.frame_mode}' mode it will be ignored.")
    
    return args

def compute_metrics(targets, predictions, num_classes):
    """
    Compute comprehensive metrics for model evaluation.
    
    Args:
        targets (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Model predictions
        num_classes (int): Number of classes
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    
    # Compute class-wise metrics
    precision = precision_score(targets, predictions, average=None, zero_division=0)
    recall = recall_score(targets, predictions, average=None, zero_division=0)
    f1 = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Compute overall metrics
    accuracy = np.mean(targets == predictions)
    macro_f1 = np.mean(f1)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    
    # Compute class-wise accuracies (useful for imbalanced datasets)
    class_accuracies = np.zeros(num_classes)
    for i in range(num_classes):
        class_mask = (targets == i)
        if np.sum(class_mask) > 0:
            class_accuracies[i] = np.mean(predictions[class_mask] == i)
    
    # Compute worst-class accuracy (helps with imbalanced datasets)
    worst_class_acc = np.min(class_accuracies)
    
    # Compute balanced accuracy (average of class-wise accuracies)
    balanced_acc = np.mean(class_accuracies)
    
    # Compute comprehensive metrics that combine multiple aspects
    # 1. Standard balanced score (accuracy + macro F1)
    balanced_score = 0.5 * accuracy + 0.5 * macro_f1
    
    # 2. Robust score (rewards models that perform well on worst class)
    robust_score = 0.4 * accuracy + 0.4 * macro_f1 + 0.2 * worst_class_acc
    
    # 3. Complete score (balances all main metrics)
    complete_score = 0.3 * accuracy + 0.3 * macro_f1 + 0.2 * balanced_acc + 0.1 * macro_precision + 0.1 * macro_recall
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy * 100,  # as percentage
        'macro_f1': macro_f1 * 100,
        'macro_precision': macro_precision * 100,
        'macro_recall': macro_recall * 100,
        'balanced_acc': balanced_acc * 100,
        'worst_class_acc': worst_class_acc * 100,
        'class_accuracies': class_accuracies * 100,
        'class_f1': f1 * 100,
        'class_precision': precision * 100,
        'class_recall': recall * 100,
        'balanced_score': balanced_score * 100,
        'robust_score': robust_score * 100,
        'complete_score': complete_score * 100
    }

def save_stage_metrics(metrics, stage_name, save_dir):
    """
    Save training metrics for a specific stage.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        stage_name (str): Name of the training stage
        save_dir (str): Directory to save metrics
    """
    # Create plots directory
    plots_dir = create_unified_plots_structure(save_dir)
    
    # Plot and save training curves
    plot_training_curves(
        metrics['train_loss'], 
        metrics['val_loss'],
        metrics['train_acc'], 
        metrics['val_acc'],
        save_dir,
        title_suffix=f" - {stage_name.title()}"
    )
    
    # Save training metrics to CSV
    save_metrics_to_csv(metrics, save_dir)
    
    # Save best model results if available
    if 'best_results' in metrics:
        # Plot confusion matrix
        save_model_results(
            metrics['best_results'],
            save_dir,
            model_name="stage_" + stage_name,
            mode=stage_name
        )

def get_scheduler(optimizer, args, stage_name=None):
    """
    Get learning rate scheduler based on arguments.
    
    Args:
        optimizer: PyTorch optimizer
        args: Command line arguments
        stage_name: Optional stage name for specific configuration
        
    Returns:
        PyTorch learning rate scheduler
    """
    if stage_name == 'stage1':
        epochs = args.stage1_epochs
    elif stage_name == 'stage2':
        epochs = args.stage2_epochs
    else:
        epochs = args.epochs
    
    if args.scheduler == 'plateau':
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3 if stage_name != 'stage2' else 5,
            verbose=True
        )
    elif args.scheduler == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

def train_stage(model, data_loaders, device, args, stage_name):
    """
    Train model for a specific stage.
    
    Args:
        model: PyTorch model
        data_loaders: Dictionary containing data loaders
        device: PyTorch device
        args: Command line arguments
        stage_name: Name of the training stage ('stage1' or 'stage2')
        
    Returns:
        dict: Dictionary containing training metrics and best model state
    """
    print(f"\n=== {stage_name.upper()}: {'Training temporal module and classifier' if stage_name == 'stage1' else 'Fine-tuning entire network'} ===")
    
    # Configure stage-specific parameters
    if stage_name == 'stage1':
        # Freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Configure optimizer for stage 1
        optimizer = optim.Adam([
            {'params': model.temporal_encoder.parameters(), 'lr': args.lr_stage1},
            {'params': model.classifier.parameters(), 'lr': args.lr_stage1}
        ], weight_decay=args.weight_decay)
        
        epochs = args.stage1_epochs
        batch_size = args.batch_size_stage1
        
    else:  # stage2
        # Unfreeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = True
        
        # Configure optimizer for stage 2 with different learning rates
        optimizer = optim.Adam([
            {'params': model.backbone.parameters(), 'lr': args.lr_stage2},
            {'params': model.temporal_encoder.parameters(), 'lr': args.lr_stage2 * 2},
            {'params': model.classifier.parameters(), 'lr': args.lr_stage2 * 2}
        ], weight_decay=args.weight_decay)
        
        epochs = args.stage2_epochs
        batch_size = args.batch_size_stage2
    
    # Get scheduler and criterion
    scheduler = get_scheduler(optimizer, args, stage_name)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0)
    
    # Get data loaders for this stage
    train_loader = data_loaders[f'train_{stage_name}']
    val_loader = data_loaders[f'val_{stage_name}']
    
    # Initialize metrics
    best_val_metric = 0.0
    best_val_epoch = 0
    best_model_state = None
    best_results = None
    
    metrics = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 'lr': [],
        'val_f1': [], 'val_balanced': []
    }
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n{stage_name.title()} - Epoch {epoch+1}/{epochs}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, targets, predictions = validate(model, val_loader, criterion, device)
        
        # Compute detailed metrics
        val_metrics = compute_metrics(targets, predictions, len(train_loader.dataset.dataset.classes))
        
        # Update metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_metrics['macro_f1'])
        metrics['val_balanced'].append(val_metrics['balanced_score'])
        
        # Update learning rate based on scheduler type
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        metrics['lr'].append(current_lr)
        
        # Print current results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_metrics['macro_f1']:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Determine if this is the best model based on selected metric
        current_metric = val_metrics[
            'accuracy' if args.model_selection == 'accuracy' else
            'macro_f1' if args.model_selection == 'f1' else
            'balanced_score'
        ]
        
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_val_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            best_results = val_metrics
            
            print(f"New best model with {args.model_selection}: {best_val_metric:.2f}%")
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{stage_name}_best.pth'))
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == epochs:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'metrics': metrics
            }, os.path.join(args.save_dir, f'{stage_name}_checkpoint_ep{epoch+1}.pth'))
    
    # Add best results to metrics
    metrics['best_epoch'] = best_val_epoch
    metrics['best_results'] = best_results
    
    # Print final results
    print(f"\n{stage_name.title()} completed.")
    print(f"Best {args.model_selection}: {best_val_metric:.2f}% at epoch {best_val_epoch}")
    
    return metrics, best_model_state

def train_full(model, data_loaders, device, args):
    """
    Train model without stage separation.
    
    Args:
        model: PyTorch model
        data_loaders: Dictionary containing data loaders
        device: PyTorch device
        args: Command line arguments
        
    Returns:
        dict: Dictionary containing training metrics and best model state
    """
    print("\n=== FULL TRAINING: Training entire network at once ===")
    
    # Configure optimizer
    optimizer = optim.Adam(
        model.parameters() if not args.freeze_backbone else model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Get scheduler and criterion
    scheduler = get_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0)
    
    # Get data loaders
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # Initialize metrics
    best_val_metric = 0.0
    best_val_epoch = 0
    best_model_state = None
    best_results = None
    
    metrics = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 'lr': [],
        'val_f1': [], 'val_balanced': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, targets, predictions = validate(model, val_loader, criterion, device)
        
        # Compute detailed metrics
        val_metrics = compute_metrics(targets, predictions, len(train_loader.dataset.classes))
        
        # Update metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_metrics['macro_f1'])
        metrics['val_balanced'].append(val_metrics['balanced_score'])
        
        # Update learning rate based on scheduler type
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        metrics['lr'].append(current_lr)
        
        # Print current results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_metrics['macro_f1']:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Determine if this is the best model based on selected metric
        current_metric = val_metrics[
            'accuracy' if args.model_selection == 'accuracy' else
            'macro_f1' if args.model_selection == 'f1' else
            'balanced_score'
        ]
        
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_val_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            best_results = val_metrics
            
            print(f"New best model with {args.model_selection}: {best_val_metric:.2f}%")
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'metrics': metrics
            }, os.path.join(args.save_dir, f'checkpoint_ep{epoch+1}.pth'))
    
    # Add best results to metrics
    metrics['best_epoch'] = best_val_epoch
    metrics['best_results'] = best_results
    
    # Print final results
    print(f"\nTraining completed.")
    print(f"Best {args.model_selection}: {best_val_metric:.2f}% at epoch {best_val_epoch}")
    
    return metrics, best_model_state

def train(args):
    """Main training function."""
    # Print system information
    print_system_info()
    
    # Create save directory and unified structure
    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = create_unified_plots_structure(args.save_dir)
    
    # Save initial configuration
    config_path = os.path.join(args.save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders
    print("\nCreating data loaders...")
    data = create_data_loaders(
        root_dir=args.data_dir,
        csv_file=args.csv_file,
        val_split=args.val_split,
        batch_size=args.batch_size,
        augment=args.augment,
        random_state=args.seed,
        frame_mode=args.frame_mode,
        num_frames=args.num_frames if hasattr(args, 'num_frames') else None,
        drop_last=True
    )
    
    train_loader = data['train']
    val_loader = data['val']
    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']
    num_classes = data['num_classes']
    
    # Print training information
    print_training_info(args, train_dataset, val_dataset, num_classes, device)
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    multi_frame_mode = args.frame_mode in ['all', 'uniform', 'key_frames']
    if multi_frame_mode:
        print(f"Using multi-frame mode: {args.frame_mode}")

    model = VideoClassifier(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=False,  # We'll handle freezing manually
        multi_frame_mode=multi_frame_mode,
        temporal_module=args.temporal_module,
        dropout_rate=args.dropout_rate,
        temporal_dropout=args.temporal_dropout,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing
    ).to(device)
    
    # Configure training mode
    if args.mode == 'freeze':
        print("Mode: Freeze backbone - Training only temporal module and classifier")
        # Freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        # Configure optimizer
        optimizer = optim.Adam([
            {'params': model.temporal_encoder.parameters(), 'lr': args.learning_rate},
            {'params': model.classifier.parameters(), 'lr': args.learning_rate}
        ], weight_decay=args.weight_decay)
        
    elif args.mode == 'finetune':
        print("Mode: Fine-tuning - Training entire network with different learning rates")
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
            
        # Configure optimizer with different learning rates
        optimizer = optim.Adam([
            {'params': model.backbone.parameters(), 'lr': args.learning_rate * args.backbone_lr_factor},
            {'params': model.temporal_encoder.parameters(), 'lr': args.learning_rate},
            {'params': model.classifier.parameters(), 'lr': args.learning_rate}
        ], weight_decay=args.weight_decay)
        
    else:  # full
        print("Mode: Full - Training entire network from scratch")
        # Configure optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # Configure scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.scheduler_factor, 
            patience=args.scheduler_patience,
            verbose=True
        )
    else:  # cosine
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_metric = 0.0
    metrics = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 'lr': [],
        'val_f1': [], 'val_balanced_score': [], 
        'val_robust_score': [], 'val_complete_score': []
    }
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                    
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    
                if 'best_val_metric' in checkpoint:
                    best_val_metric = checkpoint['best_val_metric']
                    
                print(f"Resuming from epoch {start_epoch}")
            else:
                # If it's just model weights, load them
                model.load_state_dict(checkpoint)
                print("Loaded only model weights")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Record training start time
    start_time = time.time()
    
    # Initialize variables for tracking best models
    best_val_epoch = 0
    best_model_state = None
    best_results = None
    mode_suffix = f"_{args.mode}"
    
    # Keep track of top N best models
    top_models = []  # Will contain tuples of (metric_value, epoch, model_state, results)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, targets, predictions = validate(model, val_loader, criterion, device)
        
        # Compute detailed metrics
        val_metrics = compute_metrics(targets, predictions, len(train_dataset.classes))
        
        # Update metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_metrics['macro_f1'])
        metrics['val_balanced_score'].append(val_metrics['balanced_score'])
        metrics['val_robust_score'].append(val_metrics['robust_score'])
        metrics['val_complete_score'].append(val_metrics['complete_score'])
        
        # Update learning rate based on scheduler type
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        metrics['lr'].append(current_lr)
        
        # Print current results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Metrics - F1: {val_metrics['macro_f1']:.2f}%, Robust: {val_metrics['robust_score']:.2f}%, Complete: {val_metrics['complete_score']:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Determine if this is the best model based on selected metric
        current_metric = val_metrics[args.model_selection]
        
        # Add current model to top models list
        top_models.append((current_metric, epoch + 1, model.state_dict().copy(), val_metrics))
        
        # Sort by metric value (descending) and keep top N
        top_models.sort(reverse=True)
        if len(top_models) > args.keep_best_models:
            top_models = top_models[:args.keep_best_models]
        
        # Check if current model is the best overall
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_val_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            best_results = val_metrics
            
            print(f"New best model with {args.model_selection}: {best_val_metric:.2f}%")
            
            # Save best model with mode suffix
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model{mode_suffix}.pth'))
            
            # Save best model results to plots directory
            save_model_results(
                best_results,
                args.save_dir,
                model_name=args.model_name,
                mode=args.mode,
                selection_metric=args.model_selection
            )
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint{mode_suffix}_ep{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'metrics': metrics,
                'best_val_metric': best_val_metric
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save top N models
    for i, (metric_value, epoch, model_state, model_results) in enumerate(top_models):
        # Save model
        rank = i + 1
        model_path = os.path.join(args.save_dir, f'model_rank{rank}{mode_suffix}.pth')
        torch.save(model_state, model_path)
        
        # Save metrics to plots directory
        save_model_results(
            model_results,
            args.save_dir,
            model_name=args.model_name,
            rank=rank,
            mode=args.mode,
            selection_metric=args.model_selection
        )
        
        # Create simplified model info dictionary
        model_info = {
            'rank': rank,
            'epoch': epoch,
            f'{args.model_selection}': metric_value,
            'accuracy': model_results['accuracy'],
            'macro_f1': model_results['macro_f1'],
            'balanced_score': model_results['balanced_score'],
            'robust_score': model_results['robust_score'],
            'complete_score': model_results['complete_score']
        }
        
        print(f"Saved Rank {rank} model ({args.model_selection}: {metric_value:.2f}%, epoch {epoch})")
    
    # Add best results to metrics
    metrics['best_epoch'] = best_val_epoch
    metrics['best_results'] = best_results
    
    # Save final metrics
    plot_training_curves(
        metrics['train_loss'],
        metrics['val_loss'],
        metrics['train_acc'],
        metrics['val_acc'],
        args.save_dir,
        title_suffix=f" - {args.mode.title()}"
    )
    
    # Plot advanced metrics
    plot_advanced_metrics(
        metrics,
        args.save_dir,
        mode_suffix=args.mode,
        model_name=args.model_name
    )
    
    save_metrics_to_csv(metrics, args.save_dir)
    
    # Print final timing
    total_time = time.time() - start_time
    
    # Create detailed training config
    config = {
        'model': {
            'name': args.model_name,
            'pretrained': args.pretrained,
            'num_params': sum(p.numel() for p in model.parameters()),
            'num_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'temporal_module': args.temporal_module
        },
        'training': {
            'mode': args.mode,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'optimizer': 'Adam',
            'learning_rate': args.learning_rate,
            'backbone_lr_factor': args.backbone_lr_factor if args.mode == 'finetune' else None,
            'weight_decay': args.weight_decay,
            'augmentation': args.augment,
            'frame_mode': args.frame_mode,
            'num_frames': args.num_frames if hasattr(args, 'num_frames') else None,
            'scheduler': args.scheduler
        },
        'results': {
            'best_epoch': best_val_epoch,
            'best_metric': best_val_metric,
            'selection_criterion': args.model_selection,
            'total_epochs': args.epochs,
            'training_time_minutes': total_time / 60
        }
    }
    
    # Organize and save configs to config directory
    organize_config_files(
        args.save_dir,
        config_path,  # Initial config file path
        config,       # Training results config data
        args.mode
    )
    
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best {args.model_selection}: {best_val_metric:.2f}% at epoch {best_val_epoch}")
    print(f"Top {args.keep_best_models} models saved.")

def main():
    """Main function."""
    args = parse_args()
    set_seed(args.seed)
    train(args)

if __name__ == "__main__":
    main()