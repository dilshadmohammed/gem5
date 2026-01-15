#!/usr/bin/env python3
"""
Local Inference Script for BHR Anomaly Detection
================================================
This script runs on your local machine (CPU only).
It loads a pre-trained autoencoder model and detects anomalies in router traffic data.

Prerequisites:
    pip install torch pandas numpy scikit-learn matplotlib

Usage:
    python local_inference.py --data anomaly_features_p0.1.csv --model bhr_autoencoder.pth
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoencoder_model import AnomalyDetector, load_data, FEATURE_COLUMNS


def analyze_by_router(df, output_dir=None):
    """Analyze anomalies by router and identify BHR candidate."""
    
    print("\n" + "="*60)
    print("ANOMALY ANALYSIS BY ROUTER")
    print("="*60)
    
    # Group by router
    router_stats = df.groupby('router_id').agg({
        'anomaly': ['sum', 'mean', 'count'],
        'anomaly_score': ['mean', 'max', 'std'],
        'flit_in': 'mean',
        'credit_sends': 'mean'
    }).round(4)
    
    router_stats.columns = ['anomaly_count', 'anomaly_rate', 'sample_count',
                           'avg_score', 'max_score', 'std_score',
                           'avg_flit_in', 'avg_credit_sends']
    
    # Sort by anomaly rate
    router_stats_sorted = router_stats.sort_values('anomaly_rate', ascending=False)
    
    print("\nRouter Statistics (sorted by anomaly rate):")
    print(router_stats_sorted.to_string())
    
    # Identify potential BHR
    if router_stats_sorted['anomaly_rate'].iloc[0] > 0.5:
        suspect_router = router_stats_sorted.index[0]
        print(f"\n⚠️  POTENTIAL BHR DETECTED: Router {suspect_router}")
        print(f"   Anomaly Rate: {router_stats_sorted['anomaly_rate'].iloc[0]:.2%}")
        print(f"   Avg Flit In: {router_stats_sorted['avg_flit_in'].iloc[0]:.2f}")
        print(f"   Avg Credit Sends: {router_stats_sorted['avg_credit_sends'].iloc[0]:.2f}")
    else:
        print("\n✓ No clear BHR pattern detected")
    
    # Plot results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Anomaly rate by router
        ax = axes[0, 0]
        router_stats_sorted['anomaly_rate'].plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Anomaly Rate by Router')
        ax.set_xlabel('Router ID')
        ax.set_ylabel('Anomaly Rate')
        ax.axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
        ax.legend()
        
        # Avg anomaly score by router
        ax = axes[0, 1]
        router_stats_sorted['avg_score'].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Average Anomaly Score by Router')
        ax.set_xlabel('Router ID')
        ax.set_ylabel('Avg Score')
        
        # Flit In comparison
        ax = axes[1, 0]
        router_stats_sorted['avg_flit_in'].plot(kind='bar', ax=ax, color='green')
        ax.set_title('Average Flit In by Router')
        ax.set_xlabel('Router ID')
        ax.set_ylabel('Avg Flit In')
        
        # Credit Sends comparison
        ax = axes[1, 1]
        router_stats_sorted['avg_credit_sends'].plot(kind='bar', ax=ax, color='purple')
        ax.set_title('Average Credit Sends by Router')
        ax.set_xlabel('Router ID')
        ax.set_ylabel('Avg Credit Sends')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'anomaly_analysis.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\n📊 Analysis plot saved to: {plot_path}")
        plt.close()
    
    return router_stats_sorted


def main():
    parser = argparse.ArgumentParser(description='BHR Anomaly Detection - Local Inference')
    parser.add_argument('--data', '-d', required=True, 
                       help='Path to CSV data file (anomaly_features_*.csv)')
    parser.add_argument('--model', '-m', default='bhr_autoencoder.pth',
                       help='Path to trained model file (default: bhr_autoencoder.pth)')
    parser.add_argument('--output', '-o', default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--threshold-multiplier', '-t', type=float, default=1.0,
                       help='Multiply threshold by this factor (default: 1.0)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model on Google Colab first and copy bhr_autoencoder.pth here.")
        sys.exit(1)
    
    print("="*60)
    print("BHR ANOMALY DETECTION - LOCAL INFERENCE")
    print("="*60)
    print(f"Data file: {args.data}")
    print(f"Model file: {args.model}")
    
    # Load model
    print("\n📦 Loading trained model...")
    detector = AnomalyDetector(args.model)
    
    # Adjust threshold if requested
    if args.threshold_multiplier != 1.0:
        detector.threshold *= args.threshold_multiplier
        print(f"   Adjusted threshold: {detector.threshold:.6f}")
    
    # Load data
    print("\n📊 Loading test data...")
    df = load_data(args.data)
    print(f"   Loaded {len(df)} samples from {df['router_id'].nunique()} routers")
    
    # Detect anomalies
    print("\n🔍 Detecting anomalies...")
    anomalies = detector.detect(df)
    scores = detector.detect(df, return_scores=True)
    
    df['anomaly'] = anomalies
    df['anomaly_score'] = scores
    
    total_anomalies = anomalies.sum()
    anomaly_rate = anomalies.mean()
    print(f"   Total anomalies: {total_anomalies} ({anomaly_rate:.2%})")
    
    # Analyze by router
    router_stats = analyze_by_router(df, args.output)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, 'detection_results.csv')
    df.to_csv(results_path, index=False)
    print(f"\n💾 Full results saved to: {results_path}")
    
    stats_path = os.path.join(args.output, 'router_stats.csv')
    router_stats.to_csv(stats_path)
    print(f"💾 Router statistics saved to: {stats_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
