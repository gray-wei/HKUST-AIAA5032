import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class VideoClassifier(nn.Module):
    """
    Video classifier with temporal modeling capabilities.
    """
    def __init__(
        self, 
        num_classes=10, 
        model_name='resnet50', 
        pretrained=True, 
        freeze_backbone=False,
        multi_frame_mode=False,
        temporal_module='transformer',  # ['transformer', 'lstm', 'attention']
        dropout_rate=0.5,
        temporal_dropout=0.2,
        mixup_alpha=0.2,
        label_smoothing=0.1
    ):
        """
        Initialize the video classifier.
        
        Args:
            num_classes (int): Number of output classes
            model_name (str): Name of the pre-trained model ('resnet18', 'resnet34', 'resnet50', 'densenet121', etc.)
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze the backbone network
            multi_frame_mode (bool): Whether to handle multiple frames per video
            temporal_module (str): Temporal modeling module ('transformer', 'lstm', 'attention')
            dropout_rate (float): Dropout rate for classifier
            temporal_dropout (float): Dropout rate for temporal modeling
            mixup_alpha (float): Mixup alpha parameter
            label_smoothing (float): Label smoothing parameter
        """
        super(VideoClassifier, self).__init__()
        
        self.model_name = model_name
        self.multi_frame_mode = multi_frame_mode
        self.temporal_module = temporal_module
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        
        # Load backbone CNN
        if 'resnet' in model_name:
            if model_name == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                feature_dim = self.backbone.fc.in_features
            elif model_name == 'resnet34':
                self.backbone = models.resnet34(pretrained=pretrained)
                feature_dim = self.backbone.fc.in_features
            elif model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                feature_dim = self.backbone.fc.in_features
            elif model_name == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
                feature_dim = self.backbone.fc.in_features
            else:
                raise ValueError(f"Unsupported ResNet model: {model_name}")
            
            # Replace the final fully connected layer
            self.backbone.fc = nn.Identity()
            
        elif 'densenet' in model_name:
            if model_name == 'densenet121':
                self.backbone = models.densenet121(pretrained=pretrained)
                feature_dim = self.backbone.classifier.in_features
            elif model_name == 'densenet169':
                self.backbone = models.densenet169(pretrained=pretrained)
                feature_dim = self.backbone.classifier.in_features
            else:
                raise ValueError(f"Unsupported DenseNet model: {model_name}")
            
            # Replace the classifier
            self.backbone.classifier = nn.Identity()
            
        elif 'efficientnet' in model_name:
            if model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                feature_dim = self.backbone.classifier[1].in_features
            elif model_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
                feature_dim = self.backbone.classifier[1].in_features
            else:
                raise ValueError(f"Unsupported EfficientNet model: {model_name}")
            
            # Replace the classifier
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Temporal modeling module
        if self.multi_frame_mode:
            if temporal_module == 'transformer':
                # Transformer for temporal modeling
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=feature_dim,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=temporal_dropout,
                    batch_first=True
                )
                self.temporal_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=2
                )
                self.pos_embedding = nn.Parameter(torch.randn(1, 30, feature_dim))
                
            elif temporal_module == 'lstm':
                # Bidirectional LSTM for temporal modeling
                self.temporal_encoder = nn.LSTM(
                    input_size=feature_dim,
                    hidden_size=feature_dim // 2,
                    num_layers=2,
                    bidirectional=True,
                    dropout=temporal_dropout if temporal_dropout > 0 else 0,
                    batch_first=True
                )
                
            elif temporal_module == 'attention':
                # Self-attention for temporal modeling
                self.temporal_encoder = nn.MultiheadAttention(
                    embed_dim=feature_dim,
                    num_heads=8,
                    dropout=temporal_dropout,
                    batch_first=True
                )
                self.temporal_norm = nn.LayerNorm(feature_dim)
            
            # Temporal pooling with learnable weights
            self.temporal_pool = nn.Sequential(
                nn.Linear(feature_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Classifier head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
    def forward(self, x, labels=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape:
                - Multi-frame mode: [batch_size, frames, channels, height, width]
                - Single-frame mode: [batch_size, channels, height, width]
            labels (torch.Tensor, optional): Ground truth labels
            
        Returns:
            torch.Tensor: Output predictions
        """
        if self.multi_frame_mode:
            batch_size, frame_count, channels, height, width = x.size()
            
            # 只在训练时使用mixup
            if self.training and labels is not None and self.mixup_alpha > 0:
                x, labels_a, labels_b, lam = self.mixup_data(x, labels)
            
            # Reshape to process all frames through CNN
            x = x.view(-1, channels, height, width)  # [batch_size*frame_count, channels, height, width]
            
            # Extract features for each frame
            features = self.backbone(x)  # [batch_size*frame_count, feature_dim]
            
            # Reshape to separate batch and frame dimensions
            features = features.view(batch_size, frame_count, -1)  # [batch_size, frame_count, feature_dim]
            
            # Apply temporal modeling
            if self.temporal_module == 'transformer':
                # Add positional embeddings
                features = features + self.pos_embedding[:, :frame_count, :]
                features = self.temporal_encoder(features)
                
            elif self.temporal_module == 'lstm':
                features, _ = self.temporal_encoder(features)
                
            elif self.temporal_module == 'attention':
                attn_output, _ = self.temporal_encoder(features, features, features)
                features = self.temporal_norm(features + attn_output)
            
            # Temporal pooling with learnable weights
            attention_weights = self.temporal_pool(features)
            features = torch.sum(features * attention_weights, dim=1)  # [batch_size, feature_dim]
        else:
            # Standard single-frame processing
            features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        # 只在训练时且使用了mixup时才返回混合损失
        if self.training and labels is not None and self.multi_frame_mode and self.mixup_alpha > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss_a = criterion(output, labels_a)
            loss_b = criterion(output, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            return output, loss
        
        return output
    
    def mixup_data(self, x, labels):
        """Performs mixup on the input and target."""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1-lam)  # 确保混合比例不会太极端
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed_x, labels_a, labels_b, lam
    
    def get_embedding(self, x):
        """
        Get feature embeddings from the backbone.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature embeddings
        """
        if self.multi_frame_mode:
            batch_size, frame_count, channels, height, width = x.size()
            x = x.view(-1, channels, height, width)
            features = self.backbone(x)
            features = features.view(batch_size, frame_count, -1)
            
            if self.temporal_module == 'transformer':
                features = features + self.pos_embedding[:, :frame_count, :]
                features = self.temporal_encoder(features)
            elif self.temporal_module == 'lstm':
                features, _ = self.temporal_encoder(features)
            elif self.temporal_module == 'attention':
                attn_output, _ = self.temporal_encoder(features, features, features)
                features = self.temporal_norm(features + attn_output)
            
            attention_weights = self.temporal_pool(features)
            features = torch.sum(features * attention_weights, dim=1)
        else:
            features = self.backbone(x)
        return features


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved performance.
    """
    def __init__(self, models, weights=None):
        """
        Initialize the ensemble model.
        
        Args:
            models (list): List of model instances
            weights (list, optional): List of weights for each model
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            self.weights = torch.tensor(weights) / sum(weights)
    
    def forward(self, x):
        """
        Forward pass through all models in the ensemble.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Weighted sum of model outputs
        """
        outputs = []
        
        for i, model in enumerate(self.models):
            outputs.append(model(x) * self.weights[i])
            
        return sum(outputs)