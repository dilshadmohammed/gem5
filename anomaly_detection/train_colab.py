# BHR Anomaly Detection - Colab Training Notebook
# ================================================
# Run this in Google Colab to train the autoencoder model
# Then download the trained model for local inference

# Cell 1: Setup
print("Setting up environment...")

# Install required packages (if needed)
# !pip install torch pandas numpy scikit-learn matplotlib

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from google.colab import files

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 2: Model Definition
FEATURE_COLUMNS = [
    'flit_in', 'flit_out', 'avg_wait', 'max_wait', 'buffer_occ', 
    'active_vcs', 'stalls', 'credits', 'crossbar', 'io_ratio',
    'sw_in_arb', 'sw_out_arb', 'empty_vcs', 'total_wait', 
    'min_cred', 'max_cred', 'credit_sends'
]
NUM_FEATURES = len(FEATURE_COLUMNS)

class BHRAutoencoder(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, latent_dim=4):
        super(BHRAutoencoder, self).__init__()
        
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
        return self.decoder(self.encoder(x))
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            mse = torch.mean((x - self.forward(x)) ** 2, dim=1)
        return mse

print(f"Model defined with {NUM_FEATURES} input features")

# Cell 3: Upload Training Data
print("Please upload your training data (anomaly_features_p0.csv - normal traffic)")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"Uploaded: {filename}")

# Cell 4: Load and Preprocess Data
df = pd.read_csv(filename)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Loaded {len(df)} samples")
print(f"Columns: {list(df.columns)}")

X = df[FEATURE_COLUMNS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Data shape: {X_scaled.shape}")

# Cell 5: Train Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

model = BHRAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Split data
n_val = int(len(X_scaled) * 0.2)
indices = np.random.permutation(len(X_scaled))
X_train = torch.FloatTensor(X_scaled[indices[n_val:]]).to(device)
X_val = torch.FloatTensor(X_scaled[indices[:n_val]]).to(device)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Training loop
epochs = 100
batch_size = 256
history = {'train_loss': [], 'val_loss': []}
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    losses = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size]
        optimizer.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), X_val).item()
    
    history['train_loss'].append(np.mean(losses))
    history['val_loss'].append(val_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = model.state_dict().copy()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train: {np.mean(losses):.6f}, Val: {val_loss:.6f}")

model.load_state_dict(best_state)
print(f"\nBest validation loss: {best_loss:.6f}")

# Cell 6: Set Threshold
model.eval()
errors = model.get_reconstruction_error(X_train).cpu().numpy()
threshold = np.mean(errors) + 3 * np.std(errors)
print(f"Anomaly threshold: {threshold:.6f}")

# Plot error distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(errors, bins=50, alpha=0.7)
plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Error')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

# Cell 7: Save Model
save_dict = {
    'model_state': model.state_dict(),
    'scaler': scaler,
    'threshold': threshold
}
torch.save(save_dict, 'bhr_autoencoder.pth')
print("Model saved to bhr_autoencoder.pth")

# Download model
files.download('bhr_autoencoder.pth')
print("\n✅ Download the model file and place it in your gem5/anomaly_detection folder")

# Cell 8 (Optional): Test on Attack Data
print("\nOptional: Upload attack data (anomaly_features_p0.1.csv) to test detection")
try:
    uploaded2 = files.upload()
    test_file = list(uploaded2.keys())[0]
    
    df_test = pd.read_csv(test_file)
    df_test = df_test.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = scaler.transform(df_test[FEATURE_COLUMNS].values)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    model.eval()
    test_errors = model.get_reconstruction_error(X_test_t).cpu().numpy()
    anomalies = test_errors > threshold
    
    df_test['anomaly'] = anomalies
    df_test['score'] = test_errors
    
    print("\n=== Detection Results ===")
    print(df_test.groupby('router_id')['anomaly'].agg(['sum', 'mean']).round(4))
    
    # Highlight Router 10
    r10 = df_test[df_test['router_id'] == 10]
    print(f"\nRouter 10 (BHR target):")
    print(f"  Anomaly Rate: {r10['anomaly'].mean():.2%}")
    print(f"  Avg Score: {r10['score'].mean():.6f}")
except:
    print("No test file uploaded, skipping...")

print("\n🎉 Training complete! Use local_inference.py for local detection.")
