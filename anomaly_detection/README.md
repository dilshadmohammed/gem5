# BHR Anomaly Detection with Autoencoder

This directory contains the PyTorch autoencoder model for detecting Black Hole Router (BHR) attacks in Network-on-Chip simulations.

## Files

| File | Description |
|------|-------------|
| `autoencoder_model.py` | Core model and AnomalyDetector class |
| `train_colab.py` | Training script for Google Colab (copy to Colab notebook) |
| `local_inference.py` | Local CPU inference script |

## Quick Start

### 1. Training on Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Copy contents of `train_colab.py` into cells
3. Upload `anomaly_features_p0.csv` (normal traffic data)
4. Run all cells
5. Download `bhr_autoencoder.pth` when prompted

### 2. Local Inference

```bash
# Install requirements
pip install torch pandas numpy scikit-learn matplotlib

# Run detection
python local_inference.py --data anomaly_features_p0.1.csv --model bhr_autoencoder.pth
```

## Model Architecture

```
Input (17 features)
    ↓
Encoder: 17 → 12 → 8 → 4 (latent)
    ↓
Decoder: 4 → 8 → 12 → 17
    ↓
Reconstruction Error → Anomaly Score
```

## Features Used (17 total)

- `flit_in`, `flit_out` - Flit traffic
- `avg_wait`, `max_wait`, `total_wait` - Wait times
- `buffer_occ`, `active_vcs`, `empty_vcs` - Buffer/VC status
- `stalls` - Stall cycles
- `credits`, `min_cred`, `max_cred` - Credit info
- `crossbar`, `sw_in_arb`, `sw_out_arb` - Switch activity
- `io_ratio` - Input/output ratio
- `credit_sends` - Credits sent (key BHR indicator!)

## Expected Output

For BHR attack data (e.g., probability=0.1), Router 10 should show:
- Higher anomaly rate than other routers
- Higher average anomaly score
- Significantly higher `flit_in` and `credit_sends`
