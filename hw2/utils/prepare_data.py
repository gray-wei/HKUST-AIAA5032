import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import save_data_split

def visualize_class_distribution(df, title, save_path=None):
    """
    Visualize class distribution in the dataset.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Count samples per class
    class_counts = df['Category'].value_counts().sort_index()
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution - {title}')
    
    # Add counts on top of the bars
    for i, count in enumerate(class_counts.values):
        plt.text(i, count + 5, str(count), ha='center')
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Define paths
    data_dir = 'data'
    trainval_csv = os.path.join(data_dir, 'trainval.csv')
    output_dir = data_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(trainval_csv)
    
    # Print dataset information
    print(f"Total number of samples: {len(df)}")
    print(f"Number of classes: {df['Category'].nunique()}")
    print(f"Class distribution:\n{df['Category'].value_counts().sort_index()}")
    
    # Split data into train and validation sets (stratified by class)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Category']
    )
    
    # Print split information
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Visualize class distribution
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    visualize_class_distribution(
        df, 'Full Dataset', 
        save_path=os.path.join(figures_dir, 'full_dataset_class_distribution.png')
    )
    
    visualize_class_distribution(
        train_df, 'Train Set',
        save_path=os.path.join(figures_dir, 'train_class_distribution.png')
    )
    
    visualize_class_distribution(
        val_df, 'Validation Set',
        save_path=os.path.join(figures_dir, 'val_class_distribution.png')
    )
    
    # Save splits to CSV
    save_data_split(train_df, val_df, output_dir)
    
    print("\nData preparation completed!")

if __name__ == "__main__":
    main()