import torch
from tqdm import tqdm

def print_system_info():
    """Print system and GPU information."""
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # if torch.cuda.is_available():
    #     print("\n=== GPU Information ===")
    #     for i in range(torch.cuda.device_count()):
    #         gpu = torch.cuda.get_device_properties(i)
    #         print(f"GPU {i}: {gpu.name}")
    #         print(f"Memory: {gpu.total_memory / 1024**3:.1f} GB")
    #         print(f"Compute capability: {gpu.major}.{gpu.minor}")

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if model.multi_frame_mode and model.training:
            outputs, loss = model(inputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_targets, all_predictions

def print_training_info(args, train_dataset, val_dataset, num_classes, device):
    """
    Print training information.
    
    Args:
        args: Command line arguments
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes
        device: PyTorch device
    """
    print("\n---------- Training Information ----------")
    print(f"Mode: {args.mode}")
    print(f"Backbone: {args.model_name} (pretrained: {args.pretrained})")
    print(f"Temporal module: {args.temporal_module}")
    print(f"Frame mode: {args.frame_mode}")
    
    if args.frame_mode == 'uniform':
        print(f"Number of frames: {args.num_frames}")
    
    print(f"\nTraining set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    
    if args.mode == 'freeze':
        print("Training only temporal module and classifier (backbone frozen)")
    elif args.mode == 'finetune':
        print(f"Fine-tuning entire network (backbone LR factor: {args.backbone_lr_factor})")
    else:
        print("Training entire network from scratch")
    
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Scheduler: {args.scheduler}")
    
    if args.scheduler == 'plateau':
        print(f"  - patience: {args.scheduler_patience}")
        print(f"  - factor: {args.scheduler_factor}")
    
    print(f"Model selection metric: {args.model_selection}")
    print(f"Augmentation: {args.augment}")
    print(f"Device: {device}")
    print("------------------------------------------") 