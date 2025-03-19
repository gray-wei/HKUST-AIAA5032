import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

# Import custom modules
from utils.data_utils import VideoFrameDataset, get_transforms, custom_collate_fn
from models.cnn_model import VideoClassifier

def create_test_loader(root_dir, csv_file, batch_size=32, frame_mode='uniform', num_frames=6):
    """
    Create a data loader for testing.
    
    Args:
        root_dir (string): Directory with all the video frame folders.
        csv_file (string): Path to the csv file with video IDs.
        batch_size (int): Batch size.
        frame_mode (string): How to select frame from video.
        num_frames (int): Number of frames to select when mode is uniform.
        
    Returns:
        DataLoader: Test data loader.
    """
    # Get transforms (use validation transforms)
    transforms_dict = get_transforms(augment=False)
    
    # Create dataset
    test_dataset = VideoFrameDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        transform=transforms_dict['val'],
        mode=frame_mode,
        num_frames=num_frames if frame_mode == 'uniform' else None
    )
    
    # Create data loader with custom collate function
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return test_loader, test_dataset

def test(args):
    """
    Test the model on the test set and save predictions to a CSV file.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Read original test CSV to get video IDs in correct order
    test_df = pd.read_csv(args.test_csv)
    video_ids_order = test_df['Id'].tolist()
    
    # Create test data loader
    print("Creating test data loader...")
    test_loader, test_dataset = create_test_loader(
        root_dir=args.data_dir,
        csv_file=args.test_csv,
        batch_size=args.batch_size,
        frame_mode=args.frame_mode,
        num_frames=args.num_frames
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    # Check if using multi-frame mode
    multi_frame_mode = args.frame_mode in ['all', 'uniform', 'key_frames']
    if multi_frame_mode:
        print(f"Using multi-frame mode: {args.frame_mode}")

    model = VideoClassifier(
        num_classes=args.num_classes,
        model_name=args.model_name,
        pretrained=False,
        multi_frame_mode=multi_frame_mode,
        temporal_module=args.temporal_module,
        dropout_rate=args.dropout_rate,
        temporal_dropout=args.temporal_dropout,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully")
    model.eval()
    
    # Run inference
    print("Running inference on test set...")
    predictions = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for inputs, _ in progress_bar:
            # Move inputs to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
            elif isinstance(inputs, (tuple, list)):
                inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
            
            # Forward pass
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            # Collect predictions
            predictions.extend(preds.cpu().numpy().tolist())
    
    # Create results DataFrame with original video IDs
    results = pd.DataFrame({
        'Id': video_ids_order,
        'Category': predictions
    })
    
    # Save results to CSV without index
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # Print prediction distribution
    print("\nPrediction distribution:")
    print(results['Category'].value_counts().sort_index())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test video classifier")
    
    # Data related arguments
    parser.add_argument('--data_dir', type=str, default='video_frames_30fpv_320p', 
                        help='Directory with video frames')
    parser.add_argument('--test_csv', type=str, default='data/test_for_student.csv', 
                        help='CSV file with test video IDs')
    parser.add_argument('--frame_mode', type=str, default='uniform', 
                        choices=['middle', 'random', 'all', 'uniform', 'key_frames'], 
                        help='How to select frame from videos')
    parser.add_argument('--num_frames', type=int, default=6,
                        help='Number of frames to select when mode is uniform (3-30)')
    
    # Model related arguments
    parser.add_argument('--model_path', type=str, default='results/best_model.pth', 
                        help='Path to the saved model')
    parser.add_argument('--model_name', type=str, default='resnet50', 
                        help='Name of the model architecture')
    parser.add_argument('--num_classes', type=int, default=10, 
                        help='Number of output classes')
    parser.add_argument('--temporal_module', type=str, default='attention',
                        choices=['transformer', 'lstm', 'attention'],
                        help='Temporal modeling module type')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate for classifier')
    parser.add_argument('--temporal_dropout', type=float, default=0.2,
                        help='Dropout rate for temporal module')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing parameter')
    
    # Test related arguments
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for testing')
    parser.add_argument('--output_csv', type=str, default='results/predictions.csv', 
                        help='Path to save predictions CSV')
    
    args = parser.parse_args()
    
    # Validate num_frames based on frame_mode
    if args.frame_mode == 'uniform':
        if args.num_frames < 3 or args.num_frames > 30:
            parser.error("For 'uniform' mode, num_frames must be between 3 and 30")
    elif args.num_frames != 6:
        print(f"\nWarning: --num_frames is only used in 'uniform' mode. "
              f"In '{args.frame_mode}' mode it will be ignored.")
    
    # Start testing
    test(args) 