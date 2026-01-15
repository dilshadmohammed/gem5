#!/usr/bin/env python3
"""
Real-Time BHR Detection System
==============================
Automatically detects:
1. WHICH router is compromised (BHR)
2. WHEN the attack occurs (time windows/cycles)

This runs locally on CPU after training on Colab.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import argparse
import os
import sys
from datetime import datetime

# Feature columns
FEATURE_COLUMNS = [
    'flit_in', 'flit_out', 'avg_wait', 'max_wait', 'buffer_occ', 
    'active_vcs', 'stalls', 'credits', 'crossbar', 'io_ratio',
    'sw_in_arb', 'sw_out_arb', 'empty_vcs', 'total_wait', 
    'min_cred', 'max_cred', 'credit_sends'
]
NUM_FEATURES = len(FEATURE_COLUMNS)


class BHRAutoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    def __init__(self, input_dim=NUM_FEATURES, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12), nn.ReLU(), nn.BatchNorm1d(12),
            nn.Linear(12, 8), nn.ReLU(), nn.BatchNorm1d(8),
            nn.Linear(8, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(), nn.BatchNorm1d(8),
            nn.Linear(8, 12), nn.ReLU(), nn.BatchNorm1d(12),
            nn.Linear(12, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def get_error(self, x):
        with torch.no_grad():
            return torch.mean((x - self.forward(x)) ** 2, dim=1)


class RealTimeDetector:
    """
    Real-time BHR detection system.
    
    Capabilities:
    - Per-router anomaly tracking
    - Time window identification
    - Automatic BHR identification
    - Confidence scoring
    """
    
    def __init__(self, model_path):
        self.device = torch.device('cpu')  # Force CPU for local
        self.model = BHRAutoencoder().to(self.device)
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"  Threshold: {self.threshold:.6f}")
    
    def analyze_stream(self, df, window_size=100):
        """
        Analyze data as a stream of time windows.
        
        Returns:
            dict: Per-router analysis with anomaly windows
        """
        # Preprocess
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        X = df[FEATURE_COLUMNS].values
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores for all rows
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        scores = self.model.get_error(X_tensor).numpy()
        
        df = df.copy()
        df['anomaly_score'] = scores
        df['is_anomaly'] = scores > self.threshold
        
        # Analyze per router
        router_analysis = {}
        for router_id in sorted(df['router_id'].unique()):
            router_df = df[df['router_id'] == router_id].copy()
            
            analysis = {
                'router_id': router_id,
                'total_samples': len(router_df),
                'anomaly_count': router_df['is_anomaly'].sum(),
                'anomaly_rate': router_df['is_anomaly'].mean(),
                'avg_score': router_df['anomaly_score'].mean(),
                'max_score': router_df['anomaly_score'].max(),
                'anomaly_windows': self._find_anomaly_windows(router_df),
                'feature_stats': self._get_feature_stats(router_df)
            }
            router_analysis[router_id] = analysis
        
        return router_analysis
    
    def _find_anomaly_windows(self, router_df):
        """Find continuous time windows where anomalies occur."""
        windows = []
        in_window = False
        window_start = None
        
        for idx, row in router_df.iterrows():
            if row['is_anomaly'] and not in_window:
                # Start new window
                in_window = True
                window_start = row['tick']
                window_scores = [row['anomaly_score']]
            elif row['is_anomaly'] and in_window:
                # Continue window
                window_scores.append(row['anomaly_score'])
            elif not row['is_anomaly'] and in_window:
                # End window
                windows.append({
                    'start_tick': window_start,
                    'end_tick': row['tick'],
                    'duration': row['tick'] - window_start,
                    'avg_score': np.mean(window_scores),
                    'max_score': np.max(window_scores),
                    'sample_count': len(window_scores)
                })
                in_window = False
        
        # Close any open window
        if in_window:
            windows.append({
                'start_tick': window_start,
                'end_tick': router_df['tick'].iloc[-1],
                'duration': router_df['tick'].iloc[-1] - window_start,
                'avg_score': np.mean(window_scores),
                'max_score': np.max(window_scores),
                'sample_count': len(window_scores)
            })
        
        return windows
    
    def _get_feature_stats(self, router_df):
        """Get key feature statistics for a router."""
        return {
            'avg_flit_in': router_df['flit_in'].mean(),
            'avg_credit_sends': router_df['credit_sends'].mean(),
            'avg_crossbar': router_df['crossbar'].mean()
        }
    
    def identify_bhr(self, router_analysis, min_anomaly_rate=0.3):
        """
        Automatically identify which router is likely the BHR.
        
        Criteria:
        - Highest anomaly rate
        - Significantly higher than other routers
        - Anomaly rate > threshold
        
        Returns:
            tuple: (suspected_router_id, confidence, evidence)
        """
        # Sort routers by anomaly rate
        sorted_routers = sorted(
            router_analysis.items(),
            key=lambda x: x[1]['anomaly_rate'],
            reverse=True
        )
        
        if not sorted_routers:
            return None, 0, "No data"
        
        top_router_id, top_data = sorted_routers[0]
        top_rate = top_data['anomaly_rate']
        
        # Check if meets minimum threshold
        if top_rate < min_anomaly_rate:
            return None, 0, f"No router exceeds {min_anomaly_rate:.0%} anomaly rate"
        
        # Calculate how much higher than average
        all_rates = [r[1]['anomaly_rate'] for r in sorted_routers]
        avg_rate = np.mean(all_rates)
        std_rate = np.std(all_rates)
        
        if std_rate > 0:
            z_score = (top_rate - avg_rate) / std_rate
        else:
            z_score = 0
        
        # Calculate confidence (0-100%)
        confidence = min(100, max(0, 
            50 +  # Base
            min(30, z_score * 10) +  # Statistical deviation
            min(20, top_rate * 50)   # Raw anomaly rate
        ))
        
        evidence = {
            'anomaly_rate': f"{top_rate:.2%}",
            'z_score': f"{z_score:.2f}",
            'avg_flit_in': f"{top_data['feature_stats']['avg_flit_in']:.1f}",
            'anomaly_windows': len(top_data['anomaly_windows']),
            'total_anomalies': top_data['anomaly_count']
        }
        
        return top_router_id, confidence, evidence
    
    def print_report(self, router_analysis, show_windows=5):
        """Print a comprehensive detection report."""
        print("\n" + "="*70)
        print("  BHR REAL-TIME DETECTION REPORT")
        print("="*70)
        
        # Identify BHR
        bhr_id, confidence, evidence = self.identify_bhr(router_analysis)
        
        if bhr_id is not None:
            print(f"\n🚨 SUSPECTED BHR ROUTER: {bhr_id}")
            print(f"   Confidence: {confidence:.0f}%")
            print(f"   Evidence:")
            for k, v in evidence.items():
                print(f"     - {k}: {v}")
        else:
            print(f"\n✓ No BHR detected. {evidence}")
        
        # Per-router summary
        print("\n" + "-"*70)
        print("  PER-ROUTER ANOMALY SUMMARY")
        print("-"*70)
        print(f"{'Router':>8} {'Anomalies':>10} {'Rate':>8} {'Avg Score':>12} {'Windows':>8}")
        print("-"*70)
        
        for rid, data in sorted(router_analysis.items(), 
                                key=lambda x: x[1]['anomaly_rate'], 
                                reverse=True):
            marker = " 🚨" if rid == bhr_id else ""
            print(f"{rid:>8} {data['anomaly_count']:>10} {data['anomaly_rate']:>7.1%} "
                  f"{data['avg_score']:>12.4f} {len(data['anomaly_windows']):>8}{marker}")
        
        # Show anomaly windows for suspected BHR
        if bhr_id is not None:
            windows = router_analysis[bhr_id]['anomaly_windows']
            if windows:
                print("\n" + "-"*70)
                print(f"  ANOMALY TIME WINDOWS (Router {bhr_id})")
                print("-"*70)
                print(f"{'Window':>8} {'Start Tick':>15} {'End Tick':>15} {'Duration':>12} {'Max Score':>10}")
                print("-"*70)
                
                for i, w in enumerate(windows[:show_windows]):
                    print(f"{i+1:>8} {w['start_tick']:>15} {w['end_tick']:>15} "
                          f"{w['duration']:>12} {w['max_score']:>10.4f}")
                
                if len(windows) > show_windows:
                    print(f"         ... and {len(windows) - show_windows} more windows")
        
        print("\n" + "="*70)
        return bhr_id, confidence


def main():
    parser = argparse.ArgumentParser(description='Real-Time BHR Detection')
    parser.add_argument('--data', '-d', required=True, help='CSV data file')
    parser.add_argument('--model', '-m', default='bhr_autoencoder.pth', help='Model file')
    parser.add_argument('--output', '-o', help='Save results to CSV')
    parser.add_argument('--windows', '-w', type=int, default=10, help='Max windows to show')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Train the model on Colab first!")
        sys.exit(1)
    
    # Load and analyze
    detector = RealTimeDetector(args.model)
    
    print(f"\n📊 Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"   Loaded {len(df)} samples")
    
    print("\n🔍 Analyzing...")
    analysis = detector.analyze_stream(df)
    
    # Print report
    bhr_id, confidence = detector.print_report(analysis, show_windows=args.windows)
    
    # Save detailed results
    if args.output:
        results = []
        for rid, data in analysis.items():
            results.append({
                'router_id': rid,
                'anomaly_count': data['anomaly_count'],
                'anomaly_rate': data['anomaly_rate'],
                'avg_score': data['avg_score'],
                'max_score': data['max_score'],
                'is_suspected_bhr': rid == bhr_id,
                'num_windows': len(data['anomaly_windows'])
            })
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
