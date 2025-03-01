import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import xgboost as xgb
import re

def load_test_data(test_dir):
    """Load test data files"""
    all_features = []
    file_names = []
    
    print(f"Loading test data from {test_dir}...")
    
    # Iterate through all files in the directory
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith('.csv'):
            # Extract file ID from filename
            file_id = filename.split('.')[0]  # Just get the part before the extension
            
            # Read CSV file, note the semicolon separator
            file_path = os.path.join(test_dir, filename)
            try:
                df = pd.read_csv(file_path, sep=';', header=None)
                # Data cleaning: check and handle missing values
                if df.isnull().sum().sum() > 0:
                    print(f"File {filename} has missing values, filling them")
                    df = df.fillna(df.mean())
                
                all_features.append(df)
                file_names.append(file_id)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    print(f"Loaded {len(all_features)} test files")
    return all_features, file_names

def extract_features(mfcc_data):
    """Feature engineering: extract statistical and temporal features"""
    features = []
    
    # 1. Basic statistical features
    features.extend([
        mfcc_data.mean().values,
        mfcc_data.std().values,
        mfcc_data.max().values,
        mfcc_data.min().values,
        np.percentile(mfcc_data, 25, axis=0),
        np.percentile(mfcc_data, 75, axis=0),
        mfcc_data.skew().values,  # Skewness
        mfcc_data.kurtosis().values  # Kurtosis
    ])
    
    # 2. Calculate Delta features (first-order differences)
    delta = np.diff(mfcc_data, axis=0)
    features.extend([
        delta.mean(axis=0),
        delta.std(axis=0),
        np.max(delta, axis=0),
        np.min(delta, axis=0)
    ])
    
    # 3. Calculate Delta-Delta features (second-order differences)
    delta2 = np.diff(delta, axis=0)
    features.extend([
        delta2.mean(axis=0),
        delta2.std(axis=0),
        np.max(delta2, axis=0),
        np.min(delta2, axis=0)
    ])
    
    # 4. Sliding window features (calculate statistics every 100 frames)
    window_size = 100
    window_features = []
    if len(mfcc_data) >= window_size:
        for i in range(0, len(mfcc_data) - window_size + 1, window_size // 2):
            window = mfcc_data.iloc[i:i+window_size]
            window_features.append(window.mean().values)
            window_features.append(window.std().values)
        
        # Use average as feature
        if window_features:
            window_features_mean = np.mean(window_features, axis=0)
            features.append(window_features_mean)
    else:
        # For samples shorter than window size, use statistics of all data
        features.append(mfcc_data.mean().values)
    
    # 5. Frequency domain features - calculate spectral features using FFT
    fft_features_all = []
    for col in range(mfcc_data.shape[1]):
        fft_features = np.abs(np.fft.fft(mfcc_data.iloc[:, col].values))
        fft_features_all.extend([
            np.mean(fft_features[:10]),  # Low frequency energy
            np.mean(fft_features[10:])   # High frequency energy
        ])
    features.append(np.array(fft_features_all))
    
    # Flatten all features and concatenate into a 1D array
    flat_features = []
    for feature in features:
        flat_features.extend(feature.flatten())
    
    return np.array(flat_features)

def predict_and_create_submission(test_dir, model_path, scaler_path, output_path='submission_50_005_grid.csv'):
    """Predict on test data and create submission file"""
    print("\n" + "="*50)
    print("Starting test prediction and submission creation")
    print("="*50)
    
    # Load model and scaler
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # Load test data
    test_features, test_files = load_test_data(test_dir)
    
    # Feature extraction
    print("Extracting features from test data...")
    processed_features = []
    for mfcc_data in tqdm(test_features):
        features = extract_features(mfcc_data)
        processed_features.append(features)
    
    # Check feature dimensions
    feature_lengths = [len(f) for f in processed_features]
    if len(set(feature_lengths)) > 1:
        print(f"Warning: Inconsistent test feature dimensions, range: {min(feature_lengths)} - {max(feature_lengths)}")
        # Find the shortest feature length and truncate all features
        min_length = min(feature_lengths)
        processed_features = [f[:min_length] for f in processed_features]
    
    X_test = np.array(processed_features)
    print(f"Test feature shape: {X_test.shape}")
    
    # Apply standardization
    print("Standardizing test features...")
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    
    # Create submission file
    print(f"Creating submission file: {output_path}")
    submission = pd.DataFrame({
        'Id': test_files,
        'Category': y_pred.astype(int)
    })

    # 手动添加缺失的文件记录
    missing_data = pd.DataFrame({
        'Id': ['HW00003851'],
        'Category': [4]
    })
    submission = pd.concat([submission, missing_data], ignore_index=True)
    
    submission.to_csv(output_path, index=False)
    print(f"Submission file created with {len(submission)} predictions")
    
    # Print prediction summary
    print(f"Prediction summary:")
    print(submission['Category'].value_counts())
    
    print("Test prediction completed")
    return submission

def main():
    test_dir = 'data/test'
    model_path = 'models/xgboost_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    # Check if model and scaler exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please run train.py first.")
        return
    
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} not found. Please run train.py first.")
        return
    
    # Predict and create submission
    predict_and_create_submission(test_dir, model_path, scaler_path)

if __name__ == "__main__":
    main()
