"""
Autoencoder Model for BHR Anomaly Detection
=============================================
This model is designed to detect Black Hole Router attacks in NoC (Network-on-Chip).

Training: Google Colab (with GPU)
Inference: Local CPU

Features tracked (19 columns):
- tick, router_id, flit_in, flit_out, avg_wait, max_wait
- buffer_occ, active_vcs, stalls, credits, crossbar, io_ratio
- sw_in_arb, sw_out_arb, empty_vcs, total_wait, min_cred, max_cred, credit_sends
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Feature columns (excluding tick and router_id which are metadata)
FEATURE_COLUMNS = [
    'flit_in', 'flit_out', 'avg_wait', 'max_wait', 'buffer_occ', 
    'active_vcs', 'stalls', 'credits', 'crossbar', 'io_ratio',
    'sw_in_arb', 'sw_out_arb', 'empty_vcs', 'total_wait', 
    'min_cred', 'max_cred', 'credit_sends'
]

NUM_FEATURES = len(FEATURE_COLUMNS)


class BHRAutoencoder(nn.Module):
    """
    Autoencoder for BHR anomaly detection.
    
    Architecture:
    - Encoder: 17 -> 12 -> 8 -> 4 (latent space)
    - Decoder: 4 -> 8 -> 12 -> 17
    
    Anomaly detection: High reconstruction error = anomaly
    """
    
    def __init__(self, input_dim=NUM_FEATURES, latent_dim=4):
        super(BHRAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.BatchNorm1d(12),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 12),
            nn.ReLU(),
            nn.BatchNorm1d(12),
            nn.Linear(12, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error (MSE) for each sample."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


class AnomalyDetector:
    """
    Wrapper class for training and inference.
    Handles data preprocessing, model training, and anomaly detection.
    """
    
    def __init__(self, model_path=None):
        self.model = BHRAutoencoder()
        self.scaler = StandardScaler()
        self.threshold = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def preprocess(self, df):
        """Extract and normalize features from dataframe."""
        # Select feature columns
        X = df[FEATURE_COLUMNS].values
        return X
    
    def fit_scaler(self, X):
        """Fit the scaler on training data."""
        self.scaler.fit(X)
    
    def transform(self, X):
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)
    
    def train(self, train_data, epochs=100, batch_size=256, lr=0.001, 
              validation_split=0.2, early_stopping_patience=10):
        """
        Train the autoencoder on normal (non-attack) data.
        
        Args:
            train_data: DataFrame or numpy array with normal traffic data
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            validation_split: Fraction of data for validation
            early_stopping_patience: Stop if no improvement for N epochs
        
        Returns:
            dict: Training history
        """
        self.model.to(self.device)
        
        # Preprocess data
        if isinstance(train_data, pd.DataFrame):
            X = self.preprocess(train_data)
        else:
            X = train_data
        
        # Fit scaler and transform
        self.fit_scaler(X)
        X_scaled = self.transform(X)
        
        # Split into train/validation
        n_val = int(len(X_scaled) * validation_split)
        indices = np.random.permutation(len(X_scaled))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train = torch.FloatTensor(X_scaled[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X_scaled[val_idx]).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch = X_train[i:i+batch_size]
                
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss = criterion(val_output, X_val).item()
            
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_model_state)
                break
        
        # Calculate threshold from training data (mean + 3*std of reconstruction errors)
        self.model.eval()
        errors = self.model.get_reconstruction_error(X_train).cpu().numpy()
        self.threshold = np.mean(errors) + 3 * np.std(errors)
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
        
        return history
    
    def detect(self, data, return_scores=False):
        """
        Detect anomalies in data.
        
        Args:
            data: DataFrame or numpy array
            return_scores: If True, return reconstruction errors instead of labels
        
        Returns:
            numpy array: Boolean anomaly labels or reconstruction scores
        """
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocess
        if isinstance(data, pd.DataFrame):
            X = self.preprocess(data)
        else:
            X = data
        
        X_scaled = self.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get reconstruction errors
        errors = self.model.get_reconstruction_error(X_tensor).cpu().numpy()
        
        if return_scores:
            return errors
        
        # Classify as anomaly if error > threshold
        return errors > self.threshold
    
    def save(self, path):
        """Save model, scaler, and threshold."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model, scaler, and threshold."""
        # Load with CPU mapping for local inference
        save_dict = torch.load(path, map_location='cpu')
        self.model.load_state_dict(save_dict['model_state'])
        self.scaler = save_dict['scaler']
        self.threshold = save_dict['threshold']
        self.model.to(self.device)
        print(f"Model loaded from {path}")
        print(f"Using device: {self.device}")


def load_data(csv_path):
    """Load and clean data from CSV."""
    df = pd.read_csv(csv_path)
    # Remove any rows with NaN or inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# Example usage functions for Colab and local inference
def train_on_colab(normal_data_path, model_save_path='bhr_autoencoder.pth'):
    """
    Train the model on Google Colab.
    
    Usage in Colab:
        from autoencoder_model import train_on_colab
        train_on_colab('anomaly_features_p0.csv', 'bhr_autoencoder.pth')
    """
    print("Loading training data (normal traffic, probability=0)...")
    df = load_data(normal_data_path)
    print(f"Loaded {len(df)} samples")
    
    detector = AnomalyDetector()
    print(f"Using device: {detector.device}")
    
    print("\nTraining autoencoder...")
    history = detector.train(df, epochs=100, batch_size=256)
    
    detector.save(model_save_path)
    return detector, history


def detect_anomalies_local(data_path, model_path='bhr_autoencoder.pth'):
    """
    Run anomaly detection locally (CPU).
    
    Usage:
        from autoencoder_model import detect_anomalies_local
        results = detect_anomalies_local('anomaly_features_p0.1.csv')
    """
    print("Loading model...")
    detector = AnomalyDetector(model_path)
    
    print("Loading test data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} samples")
    
    print("\nDetecting anomalies...")
    anomalies = detector.detect(df)
    scores = detector.detect(df, return_scores=True)
    
    # Add results to dataframe
    df['anomaly'] = anomalies
    df['anomaly_score'] = scores
    
    # Summary by router
    print("\n=== Anomaly Summary by Router ===")
    summary = df.groupby('router_id').agg({
        'anomaly': ['sum', 'mean'],
        'anomaly_score': 'mean'
    }).round(4)
    summary.columns = ['anomaly_count', 'anomaly_rate', 'avg_score']
    print(summary)
    
    return df, summary


if __name__ == "__main__":
    # Demo usage
    print("BHR Autoencoder Anomaly Detection")
    print("=" * 40)
    print(f"Number of features: {NUM_FEATURES}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print("\nUsage:")
    print("  Training (Colab):  train_on_colab('anomaly_features_p0.csv')")
    print("  Inference (Local): detect_anomalies_local('anomaly_features_p0.1.csv')")
