# Video Classification System

A deep learning system for video classification that combines CNN feature extraction and temporal modeling capabilities to effectively process spatiotemporal information in videos.

## Algorithm Overview

### Overall Architecture

The video classification system adopts a three-stage architecture of "Feature Extraction + Temporal Modeling + Classification":

1. **Feature Extraction**: Uses pre-trained CNN networks (such as ResNet50, EfficientNet-B0, etc.) to extract deep features from video frames.
2. **Temporal Modeling**: Models video frame sequences through three optional temporal modeling modules (Transformer, LSTM, Attention).
3. **Classifier**: Uses a multi-layer fully connected network for final classification with various regularization techniques.




### Key Features

#### 1. Flexible Backbone Network Selection
Supports various pre-trained CNN models as feature extractors:
- ResNet Series (ResNet18, ResNet50)
- DenseNet Series (DenseNet121, DenseNet169)
- EfficientNet Series (EfficientNet-B0, EfficientNet-B1)

#### 2. Diverse Temporal Modeling
Provides three temporal modeling approaches:
- **Transformer**: Uses self-attention mechanism to capture long-range dependencies between frames
- **BiLSTM**: Uses bidirectional LSTM networks to capture sequential features
- **Attention**: Uses multi-head attention mechanism for weighted aggregation of video frames

#### 3. Advanced Training Techniques
- **Data Augmentation**: Random cropping, flipping, rotation, color jittering to enhance video frame diversity
- **Frame Sampling Strategy**: Supports various frame sampling methods (middle frame, random frame, uniform sampling, key frames)
- **MixUp**: Sample mixing augmentation to improve model robustness
- **Label Smoothing**: Reduces overfitting and improves generalization
- **Model Freezing and Fine-tuning**: Supports three training modes (classifier-only training, full network fine-tuning, training from scratch)

#### 4. Multiple Evaluation Metrics
Provides comprehensive evaluation metrics:
- Accuracy and Macro F1 Score
- Balanced Accuracy
- Robustness Score (considering worst-class performance)
- Complete Score (combining multiple metrics)

### Algorithm Flow

1. **Data Preparation**:
   - Video Frame Extraction: Extract multiple frames from each video
   - Data Split: Divide into training and validation sets
   - Data Augmentation: Apply image augmentation techniques

2. **Feature Extraction**:
   - Input video frames to CNN network
   - Extract deep feature representations for each frame

3. **Temporal Modeling**:
   - Transformer Mode: Add positional encoding and process through Transformer Encoder
   - LSTM Mode: Process sequence features through bidirectional LSTM
   - Attention Mode: Aggregate features through multi-head attention mechanism

4. **Feature Aggregation**:
   - Aggregate features through learnable temporal pooling
   - Generate video-level feature representations

5. **Classification**:
   - Final classification through multi-layer classifier
   - Apply batch normalization and dropout regularization

6. **Loss Calculation and Optimization**:
   - Cross-entropy loss (optional label smoothing)
   - Adam optimizer
   - Learning rate scheduling (ReduceLROnPlateau or CosineAnnealingLR)


## Project Structure

```
project/
│
├── data/                # Data directory
│   ├── trainval.csv     # Training and validation data labels
│   └── test_for_student.csv  # Test data IDs
│
├── video_frames_30fpv_320p/  # Video frames directory
│
├── models/              # Model definitions
│   └── cnn_model.py     # CNN model implementation
│
├── utils/               # Utility functions
│   ├── data_utils.py    # Data processing utilities
│   ├── training_utils.py  # Training utilities
│   └── visualization.py   # Visualization utilities
│
├── train.py             # Training script
├── test.py              # Testing script
└── README.md            # Project documentation
```

## Usage

### Training Model

Basic training command:

```bash
python train.py --model_name resnet50 --mode freeze --batch_size 256 --save_dir results/resnet50_bs256
```

Training mode selection:

```bash
# Mode 1: Freeze backbone, train only temporal module and classifier
python train.py --mode freeze --model_name resnet50

# Mode 2: Fine-tune entire network with smaller learning rate for backbone
python train.py --mode finetune --model_name resnet50 --backbone_lr_factor 0.1

# Mode 3: Train entire network from scratch
python train.py --mode full --model_name resnet50 --pretrained False
```

Temporal module selection:

```bash
# Use Transformer for temporal modeling
python train.py --temporal_module transformer

# Use LSTM for temporal modeling
python train.py --temporal_module lstm

# Use Attention for temporal modeling
python train.py --temporal_module attention
```

Frame sampling strategies:

```bash
# Use uniform sampling with 6 frames
python train.py --frame_mode uniform --num_frames 6

# Use middle frame
python train.py --frame_mode middle

# Use random frame
python train.py --frame_mode random

# Use all frames
python train.py --frame_mode all
```

Complete training example:

```bash
python train.py \
    --model_name resnet50 \
    --mode finetune \
    --frame_mode uniform \
    --num_frames 6 \
    --temporal_module transformer \
    --batch_size 256 \
    --epochs 30 \
    --learning_rate 0.001 \
    --backbone_lr_factor 0.1 \
    --scheduler cosine \
    --mixup_alpha 0.2 \
    --label_smoothing 0.1 \
    --dropout_rate 0.5 \
    --save_dir results/resnet50_transformer_finetune
```

### Testing Model

Basic testing command:

```bash
python test.py --model_path results/resnet50_bs256/best_model_freeze.pth --model_name resnet50
```

Complete testing example:

```bash
python test.py \
    --data_dir video_frames_30fpv_320p \
    --test_csv data/test_for_student.csv \
    --model_path results/resnet50_bs256/best_model_freeze.pth \
    --model_name resnet50 \
    --frame_mode uniform \
    --num_frames 6 \
    --temporal_module transformer \
    --output_csv results/predictions.csv
```

## Parameter Description

### Main Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Training mode: freeze (freeze backbone), finetune (full network), full (from scratch) | full |
| `--model_name` | Model architecture: resnet18/34/50/101, efficientnet_b0/b1, densenet121/169 | resnet50 |
| `--pretrained` | Whether to use pre-trained weights | True |
| `--frame_mode` | Frame selection mode: middle, random, all, uniform, key_frames | uniform |
| `--num_frames` | Number of frames to select in uniform mode | 6 |
| `--temporal_module` | Temporal modeling module: transformer, lstm, attention | attention |
| `--batch_size` | Batch size | 32 |
| `--epochs` | Number of training epochs | 30 |
| `--learning_rate` | Learning rate | 0.001 |
| `--backbone_lr_factor` | Learning rate factor for backbone (used in finetune mode) | 0.1 |
| `--scheduler` | Learning rate scheduler: plateau, cosine | plateau |
| `--mixup_alpha` | MixUp augmentation coefficient, 0 to disable | 0.2 |
| `--label_smoothing` | Label smoothing coefficient | 0.1 |
| `--dropout_rate` | Classifier dropout rate | 0.5 |
| `--model_selection` | Model selection metric: accuracy, macro_f1, balanced_score, robust_score, complete_score | robust_score |

### Main Testing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Model weights path | results/best_model.pth |
| `--output_csv` | Prediction results save path | results/predictions.csv |
| `--batch_size` | Test batch size | 32 |

## Performance Evaluation

The system uses multiple metrics to evaluate classification performance:

1. **Accuracy**: Proportion of correctly predicted samples
2. **Macro F1**: Average of F1 scores across all classes, suitable for imbalanced problems
3. **Balanced Accuracy**: Average of per-class accuracies
4. **Robust Score**: Combines overall accuracy, macro F1, and worst-class accuracy
5. **Complete Score**: Weighted score considering multiple metrics

The model saves the best checkpoint based on the specified selection metric (`--model_selection`). 