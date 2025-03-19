import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split

# 自定义的collate函数，确保张量可调整大小
def custom_collate_fn(batch):
    """
    自定义的collate函数，确保所有张量都是可调整大小的。
    
    Args:
        batch: 一个批次的数据
        
    Returns:
        处理后的批次数据
    """
    # 分离图像和标签
    images = []
    labels = []
    
    for item in batch:
        # 确保每个图像张量是可调整大小的
        if isinstance(item[0], torch.Tensor):
            if item[0].dim() == 4:  # 多帧情况 [frames, channels, height, width]
                images.append(item[0].clone())
            else:  # 单帧情况 [channels, height, width]
                images.append(item[0].clone())
        else:
            # 如果不是张量，尝试进行标准转换
            images.append(item[0])
        
        labels.append(item[1])
    
    # 尝试将图像堆叠在一起
    if all(isinstance(img, torch.Tensor) for img in images):
        if images[0].dim() == 4:  # 多帧情况
            # 确保所有图像具有相同的帧数
            frame_counts = [img.size(0) for img in images]
            if len(set(frame_counts)) > 1:
                # 如果帧数不一致，填充到最大帧数
                max_frames = max(frame_counts)
                for i in range(len(images)):
                    if images[i].size(0) < max_frames:
                        padding = torch.zeros(
                            (max_frames - images[i].size(0),) + images[i].size()[1:],
                            dtype=images[i].dtype,
                            device=images[i].device
                        )
                        images[i] = torch.cat([images[i], padding], dim=0)
            
            # 现在所有图像都有相同的帧数，可以安全堆叠
            images = torch.stack(images)
        else:
            # 单帧情况，直接堆叠
            images = torch.stack(images)
    
    # 处理标签
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)
    elif isinstance(labels[0], int) or isinstance(labels[0], float):
        labels = torch.tensor(labels)
    
    return images, labels

class VideoFrameDataset(Dataset):
    """
    Dataset for loading video frames for classification.
    """
    def __init__(self, root_dir, csv_file=None, df=None, transform=None, mode='middle', num_frames=None, augment=False):
        """
        Args:
            root_dir (string): Directory with all the video frame folders.
            csv_file (string): Path to the csv file with video IDs and labels.
            df (pandas.DataFrame): DataFrame containing video IDs and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): How to select frame from video ('middle', 'random', 'all', 'uniform', 'key_frames').
            num_frames (int, optional): Number of frames to select when mode is 'uniform' (3-30).
            augment (bool): Whether to apply data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.augment = augment
        
        # Either use provided DataFrame or load from CSV
        if df is not None:
            self.data = df
        else:
            self.data = pd.read_csv(csv_file)
            
        # Get class information
        self.classes = sorted(self.data['Category'].unique())
        self.num_classes = len(self.classes)
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Validate mode and num_frames
        if mode == 'uniform' and (num_frames is None or num_frames < 3 or num_frames > 30):
            raise ValueError("For 'uniform' mode, num_frames must be between 3 and 30")
    
    def __len__(self):
        return len(self.data)
    
    def _extract_key_frames(self, frames):
        """
        Extract key frames using scene detection or motion analysis.
        Currently using a simple difference-based approach.
        """
        import cv2
        
        frame_paths = [os.path.join(self.video_dir, frame) for frame in frames]
        frame_diffs = []
        prev_frame = None
        
        for frame_path in frame_paths:
            curr_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if prev_frame is not None:
                # Calculate frame difference
                diff = np.mean(np.abs(curr_frame - prev_frame))
                frame_diffs.append(diff)
            prev_frame = curr_frame
        
        # Add first frame (it's always a key frame)
        key_frame_indices = [0]
        
        # Find peaks in frame differences (potential key frames)
        threshold = np.mean(frame_diffs) + np.std(frame_diffs)
        for i, diff in enumerate(frame_diffs):
            if diff > threshold:
                key_frame_indices.append(i + 1)
        
        # If we have too few key frames, add some uniform samples
        if len(key_frame_indices) < 6:
            additional_indices = np.linspace(0, len(frames)-1, 6).astype(int)
            key_frame_indices = sorted(list(set(key_frame_indices + list(additional_indices))))
        
        # Select at most 6 key frames
        if len(key_frame_indices) > 6:
            key_frame_indices = sorted(key_frame_indices[:6])
        
        return [frames[i] for i in key_frame_indices]
    
    def __getitem__(self, idx):
        video_id = self.data.iloc[idx]['Id']
        class_id = self.data.iloc[idx]['Category']
        
        # Convert class to index
        label = self.class_to_idx[class_id]
        
        # Get video frames directory
        video_dir = os.path.join(self.root_dir, video_id)
        frames = sorted(os.listdir(video_dir))
        
        if self.mode == 'middle':
            # Use middle frame
            frame_idx = len(frames) // 2
            frame_path = os.path.join(video_dir, frames[frame_idx])
            img = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
            
        elif self.mode == 'random':
            # Use random frame
            frame_idx = np.random.randint(0, len(frames))
            frame_path = os.path.join(video_dir, frames[frame_idx])
            img = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
            
        elif self.mode == 'all':
            # Return all frames
            all_frames = []
            for frame in frames:
                frame_path = os.path.join(video_dir, frame)
                img = Image.open(frame_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                    
                all_frames.append(img)
                
            # Stack frames and ensure it's cloned to be resizable
            stacked_frames = torch.stack(all_frames).clone()
            return stacked_frames, label
        
        elif self.mode == 'uniform':
            # Select uniform frames
            indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            selected_frames = []
            
            for i in indices:
                frame_path = os.path.join(video_dir, frames[i])
                img = Image.open(frame_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                    
                selected_frames.append(img)
            
            # Stack frames and ensure it's cloned to be resizable
            stacked_frames = torch.stack(selected_frames).clone()
            return stacked_frames, label
        
        elif self.mode == 'key_frames':
            # Extract and use key frames
            self.video_dir = video_dir  # Needed for _extract_key_frames
            key_frames = self._extract_key_frames(frames)
            selected_frames = []
            
            for frame in key_frames:
                frame_path = os.path.join(video_dir, frame)
                img = Image.open(frame_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                    
                selected_frames.append(img)
            
            # Stack frames and ensure it's cloned to be resizable
            stacked_frames = torch.stack(selected_frames).clone()
            return stacked_frames, label
        
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

def get_transforms(augment=False):
    """
    Get transformations for data preprocessing and augmentation.
    Enhanced for video classification tasks.
    
    Args:
        augment (bool): Whether to apply data augmentation.
        
    Returns:
        dict: Dictionary containing transform pipelines for train and validation.
    """
    # Basic transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    if augment:
        # Enhanced transforms for training with video-specific augmentations
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Larger size for random cropping
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Less aggressive scale
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1  # Reduced hue variation for more natural look
                )
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),  # Occasional blur to simulate motion
            transforms.RandomAffine(
                degrees=10,  # Small rotation for camera movement simulation
                translate=(0.1, 0.1),  # Small translation
                scale=(0.9, 1.1),  # Subtle scale changes
                fill=0
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))  # Occlusion simulation
        ])
    else:
        # Basic transforms for training without augmentation
        train_transform = val_transform
    
    return {
        'train': train_transform,
        'val': val_transform
    }

def create_data_loaders(
    root_dir, 
    csv_file, 
    val_split=0.2, 
    batch_size=32, 
    augment=True, 
    random_state=42, 
    frame_mode='middle',
    num_frames=None,
    drop_last=True
):
    """
    Create train and validation data loaders.
    
    Args:
        root_dir (string): Directory with all the video frame folders.
        csv_file (string): Path to the csv file with video IDs and labels.
        val_split (float): Proportion of the dataset to use for validation.
        batch_size (int): Batch size for data loaders.
        augment (bool): Whether to apply data augmentation.
        random_state (int): Random seed for reproducibility.
        frame_mode (string): How to select frames from video.
        num_frames (int, optional): Number of frames to select when mode is 'uniform'.
                                  Ignored for other modes:
                                  - 'middle', 'random': single frame
                                  - 'all': all frames
                                  - 'key_frames': up to 6 frames
        drop_last (bool): Whether to drop the last incomplete batch.
        
    Returns:
        dict: Dictionary containing train and validation data loaders.
    """
    # Get transforms
    transforms_dict = get_transforms(augment)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=random_state, stratify=df['Category']
    )
    
    # Create datasets
    train_dataset = VideoFrameDataset(
        root_dir=root_dir,
        df=train_df,
        transform=transforms_dict['train'],
        mode=frame_mode,
        num_frames=num_frames if frame_mode == 'uniform' else None,
        augment=augment
    )
    
    val_dataset = VideoFrameDataset(
        root_dir=root_dir,
        df=val_df,
        transform=transforms_dict['val'],
        mode=frame_mode,
        num_frames=num_frames if frame_mode == 'uniform' else None,
        augment=False
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=drop_last
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'num_classes': train_dataset.num_classes
    }

def save_data_split(train_df, val_df, output_dir='data'):
    """
    Save train/validation split to CSV files.
    
    Args:
        train_df (pandas.DataFrame): Training data DataFrame.
        val_df (pandas.DataFrame): Validation data DataFrame.
        output_dir (string): Directory to save the CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Saved train split ({len(train_df)} samples) to {train_path}")
    print(f"Saved validation split ({len(val_df)} samples) to {val_path}") 