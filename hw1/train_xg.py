import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def load_data(data_dir):
    """Load all training data and extract labels"""
    all_features = []
    labels = []
    file_names = []
    
    print(f"Loading data from {data_dir}...")
    
    # Iterate through all files in the directory
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.csv'):
            # Extract label from filename
            label = int(filename.split('_')[1].split('.')[0])
            
            # Read CSV file, note the semicolon separator
            file_path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(file_path, sep=';', header=None)
                # Data cleaning: check and handle missing values
                if df.isnull().sum().sum() > 0:
                    print(f"File {filename} has missing values, filling them")
                    df = df.fillna(df.mean())
                
                all_features.append(df)
                labels.append(label)
                file_names.append(filename)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    print(f"Loaded {len(all_features)} files with {len(set(labels))} unique classes")
    return all_features, labels, file_names

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

def preprocess_data(data_dir):
    """Main function for data preprocessing and feature extraction"""
    # Load raw data
    all_features, labels, file_names = load_data(data_dir)
    
    # Feature extraction
    print("Extracting features...")
    processed_features = []
    for mfcc_data in tqdm(all_features):
        features = extract_features(mfcc_data)
        processed_features.append(features)
    
    # Check if feature dimensions are consistent
    feature_lengths = [len(f) for f in processed_features]
    if len(set(feature_lengths)) > 1:
        print(f"Warning: Inconsistent feature dimensions, range: {min(feature_lengths)} - {max(feature_lengths)}")
        # Find the shortest feature length and truncate all features
        min_length = min(feature_lengths)
        processed_features = [f[:min_length] for f in processed_features]
    
    X = np.array(processed_features)
    y = np.array(labels)
    
    print(f"Feature shape: {X.shape}, Labels shape: {y.shape}")
    
    # Data standardization
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # Save feature and label mapping
    feature_label_map = pd.DataFrame({
        'filename': file_names,
        'label': labels
    })
    feature_label_map.to_csv('models/feature_label_map.csv', index=False)
    print("Feature-label mapping saved to models/feature_label_map.csv")
    
    print("Preprocessing completed successfully")
    return X_scaled, y

def visualize_features(X, y):
    """Feature visualization"""
    print("Starting feature visualization...")
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. PCA dimensionality reduction for visualization
    from sklearn.decomposition import PCA
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter, label='Class')
    plt.title('PCA Dimensionality Reduction')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig('visualizations/pca_visualization.png')
    plt.close()
    print("PCA visualization saved")
    
    # 2. t-SNE dimensionality reduction for visualization
    from sklearn.manifold import TSNE
    print("Performing t-SNE dimensionality reduction (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Dimensionality Reduction')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.savefig('visualizations/tsne_visualization.png')
    plt.close()
    print("t-SNE visualization saved")
    
    # 3. Class distribution visualization
    print("Creating class distribution visualization...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.savefig('visualizations/class_distribution.png')
    plt.close()
    print("Class distribution visualization saved")
    
    print("Feature visualization completed")

def train_model(X, y):
    """Train XGBoost model and perform cross-validation"""
    print("\n" + "="*50)
    print("Starting model training...")
    print("="*50)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Define XGBoost model parameters
    params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': len(np.unique(y)),
        'random_state': 42,
        'verbosity': 1  # Enable verbose output
    }
    
    # Define XGBoost model
    model = xgb.XGBClassifier(**params)
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train model on the full training set
    print("\nTraining final model on the full training set...")
    
    # For XGBoost 2.1.4, let's use the most basic approach first
    print("Using basic fit method for XGBoost 2.1.4...")
    model.fit(X_train, y_train)
    
    # Evaluate test set performance
    y_pred = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print(f"Test set accuracy: {test_score:.4f}")
    
    # Output detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix visualization
    print("Creating confusion matrix visualization...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved")
    
    # Feature importance visualization
    print("Creating feature importance visualization...")
    feature_importance = model.feature_importances_
    # Make sure we don't try to show more features than we have
    n_features_to_show = min(20, len(feature_importance))
    sorted_idx = np.argsort(feature_importance)[-n_features_to_show:]
    plt.figure(figsize=(10, 12))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
    plt.title(f'Feature Importance (Top {n_features_to_show})')
    plt.savefig('visualizations/feature_importance.png')
    plt.close()
    print("Feature importance visualization saved")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    try:
        # For XGBoost 2.1.4, use joblib to save the model
        joblib.dump(model, 'models/xgboost_model.pkl')
        print("Model saved to models/xgboost_model.pkl")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("\nModel training completed")
    return model, test_score

def predict_test_data(model, scaler, test_dir):
    """Predict on test data"""
    print("\n" + "="*50)
    print(f"Starting prediction on test data from {test_dir}...")
    print("="*50)
    
    # Load test data
    test_features, _, test_files = load_data(test_dir)
    
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
        X_test = np.array([f[:min_length] for f in processed_features])
    else:
        X_test = np.array(processed_features)
    
    print(f"Test feature shape: {X_test.shape}")
    
    # Apply standardization
    print("Standardizing test features...")
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    # Save prediction results
    results = pd.DataFrame({
        'filename': test_files,
        'predicted_label': y_pred
    })
    
    # Add probability for each class
    for i in range(y_prob.shape[1]):
        results[f'prob_class_{i}'] = y_prob[:, i]
    
    results.to_csv('models/test_predictions.csv', index=False)
    print(f"Test predictions saved to models/test_predictions.csv")
    
    # Print prediction summary
    print(f"Prediction summary:")
    print(results['predicted_label'].value_counts())
    
    print("Test prediction completed")
    return results

def main():
    train_dir = 'data/trainval'
    test_dir = 'data/test'
    
    print("\n" + "="*50)
    print("MFCC-based Video Classification")
    print("="*50)
    
    # Data preprocessing and feature extraction
    X, y = preprocess_data(train_dir)
    
    # Feature visualization
    visualize_features(X, y)
    
    # Train model
    model, test_score = train_model(X, y)
    
    # Load saved scaler
    scaler = joblib.load('models/scaler.pkl')
    
    # Predict on test set
    if os.path.exists(test_dir):
        predict_test_data(model, scaler, test_dir)
    else:
        print(f"Test directory {test_dir} does not exist, skipping test prediction")
    
    print("\n" + "="*50)
    print("Classification pipeline completed successfully")
    print("="*50)

if __name__ == "__main__":
    main()
