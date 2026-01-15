#!/usr/bin/env python3
"""
Multi-BHR Real-Time Detection (Optimized)
==========================================
Fast per-window detection using batch processing + vectorized operations.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import os
import sys

FEATURE_COLUMNS = [
    'flit_in', 'flit_out', 'avg_wait', 'max_wait', 'buffer_occ', 
    'active_vcs', 'stalls', 'credits', 'crossbar', 'io_ratio',
    'sw_in_arb', 'sw_out_arb', 'empty_vcs', 'total_wait', 
    'min_cred', 'max_cred', 'credit_sends'
]


class BHRAutoencoder(nn.Module):
    def __init__(self, input_dim=17, latent_dim=4):
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
    def forward(self, x): return self.decoder(self.encoder(x))
    def get_error(self, x):
        with torch.no_grad(): return torch.mean((x - self.forward(x)) ** 2, dim=1)


class MultiBHRDetector:
    def __init__(self, model_path, bhr_threshold=0.3, z_threshold=1.5):
        self.device = torch.device('cpu')
        self.model = BHRAutoencoder().to(self.device)
        self.bhr_threshold = bhr_threshold
        self.z_threshold = z_threshold
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']
        self.model.eval()
        
        print(f"✓ Model loaded (threshold: {self.threshold:.4f})")
    
    def detect(self, df, window_cycles=100):
        """Fast detection using 100-cycle windows (1 cycle = 500 ticks)."""
        TICKS_PER_CYCLE = 500
        WINDOW_TICKS = window_cycles * TICKS_PER_CYCLE  # 100 cycles = 50,000 ticks
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
        df = df.sort_values(['tick', 'router_id']).reset_index(drop=True)
        
        unique_routers = sorted(df['router_id'].unique())
        n_routers = len(unique_routers)
        
        # Step 1: Batch compute ALL anomaly scores (FAST)
        X = self.scaler.transform(df[FEATURE_COLUMNS].values)
        scores = self.model.get_error(torch.FloatTensor(X)).numpy()
        df['anomaly_score'] = scores
        df['is_anomaly'] = (scores > self.threshold).astype(int)
        
        # Step 2: Bin data into 100-cycle windows
        min_tick = df['tick'].min()
        df['window_id'] = ((df['tick'] - min_tick) // WINDOW_TICKS).astype(int)
        df['window_cycle'] = df['window_id'] * window_cycles
        
        unique_windows = sorted(df['window_id'].unique())
        n_windows = len(unique_windows)
        print(f"   Using {window_cycles}-cycle windows ({n_windows} windows total)")
        
        # Step 3: Per-window statistics
        pivot_anomaly = df.pivot_table(index='window_id', columns='router_id', 
                                       values='is_anomaly', aggfunc='sum', fill_value=0)
        pivot_count = df.pivot_table(index='window_id', columns='router_id', 
                                     values='is_anomaly', aggfunc='count', fill_value=0)
        pivot_scores = df.pivot_table(index='window_id', columns='router_id',
                                      values='anomaly_score', aggfunc='mean', fill_value=0)
        
        # Cumulative sums
        cum_anomaly = pivot_anomaly.cumsum().values  # (n_windows, n_routers)
        cum_count = pivot_count.cumsum().values
        
        # Cumulative rates per window
        cum_rates = np.divide(cum_anomaly, cum_count, where=cum_count > 0, out=np.zeros_like(cum_anomaly, dtype=float))
        
        # Step 3: Identify BHRs at each window using z-score
        
        # All windows for CSV
        all_window_bhrs = []
        for idx in range(n_windows):
            rates = cum_rates[idx]
            mean_rate = np.mean(rates)
            std_rate = np.std(rates) + 1e-8
            z_scores = (rates - mean_rate) / std_rate
            
            bhrs_at_window = []
            for i, rid in enumerate(unique_routers):
                if rates[i] > self.bhr_threshold and z_scores[i] > self.z_threshold:
                    bhrs_at_window.append(int(rid))
            
            all_window_bhrs.append({
                'window_id': unique_windows[idx],
                'cycle': unique_windows[idx] * window_cycles,
                'detected_bhrs': bhrs_at_window
            })
        
        # Sampled windows for terminal display (~20 samples)
        sample_indices = list(range(0, n_windows, max(1, n_windows // 20)))
        window_bhrs = [all_window_bhrs[idx] for idx in sample_indices if idx < len(all_window_bhrs)]
        
        # Final detection
        final_rates = cum_rates[-1]
        mean_rate = np.mean(final_rates)
        std_rate = np.std(final_rates) + 1e-8
        z_scores = (final_rates - mean_rate) / std_rate
        
        detected_bhrs = []
        router_stats = []
        for i, rid in enumerate(unique_routers):
            is_bhr = final_rates[i] > self.bhr_threshold and z_scores[i] > self.z_threshold
            if is_bhr:
                detected_bhrs.append({'router': rid, 'rate': final_rates[i], 'z_score': z_scores[i]})
            router_stats.append({
                'router_id': rid,
                'anomaly_rate': final_rates[i],
                'z_score': z_scores[i],
                'is_bhr': is_bhr
            })
        
        print("   ✓ Done!")
        
        return {
            'detected_bhrs': detected_bhrs,
            'router_stats': pd.DataFrame(router_stats),
            'window_bhrs': window_bhrs,  # Sampled for terminal
            'all_window_bhrs': all_window_bhrs,  # All windows for CSV
            'heatmap_data': pivot_scores.values.T,
            'unique_routers': unique_routers,
            'unique_windows': unique_windows,
            'cum_rates': cum_rates,
            'window_cycles': window_cycles
        }
    
    def print_report(self, results):
        detected = results['detected_bhrs']
        stats = results['router_stats']
        window_bhrs = results['window_bhrs']
        
        print("\n" + "="*70)
        print("  BHR ANOMALY DETECTION RESULTS")
        print("="*70)
        
        if detected:
            print(f"\n DETECTED {len(detected)} BHR ROUTER(S):")
            for bhr in sorted(detected, key=lambda x: x['rate'], reverse=True):
                print(f"   * Router {int(bhr['router']):2d}: Rate = {bhr['rate']:.1%}, Z-Score = {bhr['z_score']:.2f}")
        else:
            print("\n No BHR routers detected")
        
        # Per-window timeline
        print("\n" + "-"*70)
        print("BHR DETECTION TIMELINE (100-cycle windows)")
        print("-"*70)
        print(f"{'Window':>8} {'Cycle':>15} {'Detected BHRs'}")
        print("-"*70)
        
        for w in window_bhrs:  # Show all sampled windows
            bhrs_str = str(w['detected_bhrs']) if w['detected_bhrs'] else "-"
            print(f"{w['window_id']:>8} {w['cycle']:>15} {bhrs_str}")
        
        # Final summary
        print("\n" + "-"*70)
        print("  FINAL PER-ROUTER SUMMARY")
        print("-"*70)
        print(f"{'Router':>8} {'Anomaly Rate':>14} {'Z-Score':>10} {'Status':>10}")
        print("-"*70)
        
        for _, row in stats.sort_values('anomaly_rate', ascending=False).iterrows():
            status = "* BHR" if row['is_bhr'] else ""
            print(f"{int(row['router_id']):>8} {row['anomaly_rate']:>13.1%} {row['z_score']:>10.2f} {status:>10}")
        
        return [b['router'] for b in detected]
    
    def generate_heatmap(self, results, output_dir='results', mesh_rows=4):
        os.makedirs(output_dir, exist_ok=True)
        
        unique_routers = results['unique_routers']
        bhr_ids = [int(b['router']) for b in results['detected_bhrs']]
        stats = results['router_stats']
        
        # ============================================
        # 1. MESH TOPOLOGY HEATMAP (4x4 grid)
        # ============================================
        # Router layout: row 0 at bottom, column 0 at left
        # Router ID = row * cols + col
        mesh_cols = len(unique_routers) // mesh_rows
        
        # Create grid with anomaly rates
        mesh_grid = np.zeros((mesh_rows, mesh_cols))
        for _, row in stats.iterrows():
            rid = int(row['router_id'])
            r = rid // mesh_cols  # row
            c = rid % mesh_cols   # col
            mesh_grid[r, c] = row['anomaly_rate']
        
        # Flip vertically so row 0 is at bottom
        mesh_grid = np.flipud(mesh_grid)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        try:
            import seaborn as sns
            sns.heatmap(mesh_grid, annot=True, fmt='.0%', cmap='YlOrRd', 
                       vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Anomaly Rate'},
                       linewidths=2, linecolor='white', square=True)
        except ImportError:
            im = ax.imshow(mesh_grid, cmap='YlOrRd', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Anomaly Rate')
            # Add text annotations
            for i in range(mesh_rows):
                for j in range(mesh_cols):
                    ax.text(j, i, f'{mesh_grid[i,j]:.0%}', ha='center', va='center', fontsize=12)
        
        # Create labels (flipped to match grid)
        row_labels = [f'Row {mesh_rows-1-i}' for i in range(mesh_rows)]
        col_labels = [f'Col {i}' for i in range(mesh_cols)]
        ax.set_yticklabels(row_labels)
        ax.set_xticklabels(col_labels)
        
        # Mark BHR routers with red border
        for rid in bhr_ids:
            r = rid // mesh_cols
            c = rid % mesh_cols
            r_flipped = mesh_rows - 1 - r  # flip for display
            rect = plt.Rectangle((c, r_flipped), 1, 1, fill=False, 
                                 edgecolor='blue', linewidth=4)
            ax.add_patch(rect)
        
        # Add router IDs as secondary labels
        for i in range(mesh_rows):
            for j in range(mesh_cols):
                rid = (mesh_rows - 1 - i) * mesh_cols + j
                bhr_marker = " *" if rid in bhr_ids else ""
                ax.text(j + 0.5, i + 0.15, f'R{rid}{bhr_marker}', 
                       ha='center', va='top', fontsize=9, color='black', fontweight='bold')
        
        ax.set_title(f'4x4 Mesh Topology - Anomaly Rates\nDetected BHRs: {bhr_ids} (blue border)', fontsize=14)
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'mesh_topology_heatmap.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\nMesh Topology Heatmap saved: {path}")
        
        # ============================================
        # 2. BAR CHART
        # ============================================
        stats_sorted = stats.sort_values('router_id', ascending=True)
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['red' if int(rid) in bhr_ids else 'steelblue' for rid in stats_sorted['router_id']]
        ax.bar(range(len(stats_sorted)), stats_sorted['anomaly_rate'], color=colors)
        ax.set_xticks(range(len(stats_sorted)))
        ax.set_xticklabels([f'R{int(r)}' for r in stats_sorted['router_id']])
        ax.set_ylabel('Anomaly Rate')
        ax.set_title(f'Anomaly Rate by Router (Detected BHRs: {bhr_ids})')
        ax.axhline(y=self.bhr_threshold, color='gray', linestyle='--', label=f'Threshold={self.bhr_threshold:.0%}')
        ax.legend()
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'anomaly_rate_bar.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Bar chart saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='BHR Anomaly Detection')
    parser.add_argument('--data', '-d', required=True, help='CSV file')
    parser.add_argument('--model', '-m', default='bhr_autoencoder.pth', help='Model file')
    parser.add_argument('--output', '-o', default='results', help='Output dir')
    parser.add_argument('--bhr-threshold', type=float, default=0.3)
    parser.add_argument('--z-threshold', type=float, default=1.5)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)
    
    print("="*70)
    print("  BHR ANOMALY DETECTION")
    print("="*70)
    
    detector = MultiBHRDetector(args.model, args.bhr_threshold, args.z_threshold)
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples")
    
    results = detector.detect(df)
    bhr_ids = detector.print_report(results)
    detector.generate_heatmap(results, args.output)
    
    results['router_stats'].to_csv(os.path.join(args.output, 'detection_summary.csv'), index=False)
    
    # Save ALL per-window BHR detections to CSV
    window_df = pd.DataFrame([{
        'window_id': w['window_id'],
        'cycle': w['cycle'],
        'detected_bhrs': str(w['detected_bhrs'])
    } for w in results['all_window_bhrs']])
    window_df.to_csv(os.path.join(args.output, 'per_window_bhrs.csv'), index=False)
    print(f"   Saved {len(window_df)} windows to per_window_bhrs.csv")
    
    print(f"\nResults saved to: {args.output}/")
    print("="*70)
    bhr_ids_int = [int(b) for b in bhr_ids]
    print(f"  COMPLETE - Detected BHRs: {bhr_ids_int}")
    print("="*70)


if __name__ == "__main__":
    main()
