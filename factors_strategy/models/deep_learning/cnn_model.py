"""
CNN Model for Stock Prediction
Convolutional Neural Network for extracting patterns from tick data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from torch.utils.data import Dataset, DataLoader
import pandas as pd

logger = logging.getLogger(__name__)


class TickDataset(Dataset):
    """Dataset for tick data with order book snapshots"""
    
    def __init__(self, 
                 tick_data: pd.DataFrame,
                 order_book_data: pd.DataFrame,
                 labels: pd.Series,
                 sequence_length: int = 100,
                 feature_cols: List[str] = None):
        """Initialize dataset"""
        self.tick_data = tick_data
        self.order_book_data = order_book_data
        self.labels = labels
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols or self._get_default_features()
        
        # Prepare data
        self._prepare_data()
        
    def _get_default_features(self) -> List[str]:
        """Get default feature columns"""
        return [
            'price', 'volume', 'spread', 'mid_price',
            'bid_volume_1', 'ask_volume_1', 'order_imbalance'
        ]
        
    def _prepare_data(self):
        """Prepare data for training"""
        # Merge tick and order book data
        self.data = pd.merge_asof(
            self.tick_data,
            self.order_book_data,
            on='timestamp',
            by='symbol',
            direction='backward'
        )
        
        # Calculate additional features
        self.data['spread'] = self.data['ask_price_1'] - self.data['bid_price_1']
        self.data['mid_price'] = (self.data['ask_price_1'] + self.data['bid_price_1']) / 2
        self.data['order_imbalance'] = (
            self.data['bid_volume_1'] - self.data['ask_volume_1']
        ) / (self.data['bid_volume_1'] + self.data['ask_volume_1'] + 1e-10)
        
        # Group by symbol for sequence generation
        self.symbol_groups = self.data.groupby('symbol')
        self.symbols = list(self.symbol_groups.groups.keys())
        
    def __len__(self) -> int:
        """Get dataset length"""
        total_sequences = 0
        for symbol in self.symbols:
            symbol_data = self.symbol_groups.get_group(symbol)
            total_sequences += max(0, len(symbol_data) - self.sequence_length)
        return total_sequences
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        # Find which symbol and position this index corresponds to
        current_idx = 0
        for symbol in self.symbols:
            symbol_data = self.symbol_groups.get_group(symbol)
            symbol_sequences = max(0, len(symbol_data) - self.sequence_length)
            
            if current_idx + symbol_sequences > idx:
                # This is the symbol
                position = idx - current_idx
                
                # Extract sequence
                sequence_data = symbol_data.iloc[position:position + self.sequence_length]
                features = sequence_data[self.feature_cols].values
                
                # Normalize features
                features = self._normalize_features(features)
                
                # Get label
                label_idx = sequence_data.index[-1]
                label = self.labels.loc[label_idx] if label_idx in self.labels.index else 0
                
                return torch.FloatTensor(features), torch.FloatTensor([label])
                
            current_idx += symbol_sequences
            
        raise IndexError("Index out of range")
        
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features"""
        # Z-score normalization per feature
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-10
        return (features - mean) / std


class TickCNN(nn.Module):
    """CNN model for tick data pattern recognition"""
    
    def __init__(self, config: Dict):
        """Initialize CNN model"""
        super(TickCNN, self).__init__()
        
        self.config = config
        self.input_channels = config['architecture']['input_channels']
        self.conv_layers = config['architecture']['conv_layers']
        self.dropout_rate = config['architecture']['dropout_rate']
        self.output_dim = config['architecture']['output_dim']
        
        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = 1  # Start with 1 channel (will reshape input)
        
        for layer_config in self.conv_layers:
            conv_block = self._build_conv_block(
                in_channels,
                layer_config['filters'],
                layer_config['kernel_size'],
                layer_config['stride'],
                layer_config['padding']
            )
            self.conv_blocks.append(conv_block)
            in_channels = layer_config['filters']
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.output_dim),
            nn.ReLU()
        )
        
        # Output layer for binary classification
        self.output_layer = nn.Sequential(
            nn.Linear(self.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _build_conv_block(self, in_channels: int, out_channels: int,
                         kernel_size: int, stride: int, padding: int) -> nn.Module:
        """Build convolutional block"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Input shape: (batch_size, sequence_length, features)
        batch_size = x.size(0)
        
        # Reshape to (batch_size, 1, sequence_length * features)
        x = x.view(batch_size, 1, -1)
        
        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        # Global pooling
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Get prediction
        output = self.output_layer(features)
        
        return output, features
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without final classification"""
        with torch.no_grad():
            _, features = self.forward(x)
        return features


class CNNTrainer:
    """Trainer for CNN model"""
    
    def __init__(self, model: TickCNN, config: Dict, device: str = 'cuda'):
        """Initialize trainer"""
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Setup loss function
        self.criterion = nn.BCELoss()
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                predicted = (output > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Train model"""
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info("Early stopping triggered")
                break
                
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint from {filepath}")
        
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        self.model.eval()
        predictions = []
        features_list = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output, features = self.model(data)
                predictions.append(output.cpu().numpy())
                features_list.append(features.cpu().numpy())
                
        predictions = np.concatenate(predictions)
        features = np.concatenate(features_list)
        
        return predictions, features


def create_cnn_model(config: Dict) -> TickCNN:
    """Factory function to create CNN model"""
    return TickCNN(config['models']['cnn'])


def prepare_training_data(tick_data: pd.DataFrame,
                         order_book_data: pd.DataFrame,
                         labels: pd.Series,
                         config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Prepare data loaders for training"""
    
    # Create dataset
    dataset = TickDataset(
        tick_data,
        order_book_data,
        labels,
        sequence_length=config['models']['cnn']['training']['sequence_length']
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['models']['cnn']['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['models']['cnn']['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader