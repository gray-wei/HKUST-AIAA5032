import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import create_data_loaders

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensors for visualization.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor.
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.
        
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    # Clone tensor to avoid modifying the original
    tensor = tensor.clone()
    
    # Denormalize
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor

def visualize_batch(images, labels, class_names=None, max_images=16, figsize=(12, 12)):
    """
    Visualize a batch of images with their labels.
    
    Args:
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Batch of labels.
        class_names (list, optional): List of class names.
        max_images (int): Maximum number of images to display.
        figsize (tuple): Figure size.
    """
    # Limit number of images to display
    batch_size = min(len(images), max_images)
    images = images[:batch_size]
    labels = labels[:batch_size]
    
    # Denormalize images
    images = denormalize(images)
    
    # Create a grid of images
    grid = make_grid(images, nrow=4, padding=4)
    
    # Convert to numpy array and transpose dimensions
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    # Plot grid
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.axis('off')
    
    # Add labels as title
    if class_names:
        label_names = [class_names[label.item()] for label in labels]
        plt.title(f"Labels: {', '.join(label_names)}")
    else:
        plt.title(f"Labels: {labels.tolist()}")
    
    plt.tight_layout()
    plt.show()

def test_dataloader():
    """Test data loader and visualize a batch of images."""
    # Define paths and parameters
    video_dir = 'video_frames_30fpv_320p'
    csv_file = 'data/trainval.csv'
    batch_size = 16
    val_split = 0.2
    augment = True
    
    # Create data loaders
    print("Creating data loaders...")
    data = create_data_loaders(
        root_dir=video_dir,
        csv_file=csv_file,
        val_split=val_split,
        batch_size=batch_size,
        augment=augment
    )
    
    train_loader = data['train']
    val_loader = data['val']
    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']
    num_classes = data['num_classes']
    
    # Print dataset information
    print(f"Number of classes: {num_classes}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Total dataset size: {len(train_dataset) + len(val_dataset)}")
    
    # Get class names (indices)
    class_names = list(train_dataset.class_to_idx.keys())
    print(f"Class names: {class_names}")
    
    # Get a batch of training data
    print("Getting a batch of training data...")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels.tolist()}")
        
        # Visualize the batch
        print("Visualizing the batch...")
        visualize_batch(images, labels, class_names=None)
        
        # Only visualize one batch
        break
    
    # Test validation loader
    print("\nGetting a batch of validation data...")
    for images, labels in val_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Visualize the batch
        print("Visualizing the batch...")
        visualize_batch(images, labels, class_names=None)
        
        # Only visualize one batch
        break
    
    print("\nDataloader test completed successfully!")

if __name__ == "__main__":
    test_dataloader() 