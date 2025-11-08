import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_stress_test_results(metrics_path: str, output_folder: str = 'graphs'):
    """
    Create comprehensive graphs from stress test metrics.
    """
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    batch_numbers = metrics['batch_numbers']
    batch_times = metrics['batch_times']
    degradation_events = metrics['degradation_events']
    baseline_avg = metrics.get('baseline_avg')
    total_transactions = metrics['total_transactions']
    
    # Calculate cumulative transactions
    cumulative_transactions = [b * 50 for b in batch_numbers]
    
    # 1. Batch Response Time Over Time
    plt.figure(figsize=(14, 6))
    plt.plot(batch_numbers, batch_times, linewidth=1, alpha=0.7, label='Batch Time')
    
    # Plot baseline
    if baseline_avg:
        plt.axhline(y=baseline_avg, color='green', linestyle='--', linewidth=2, label=f'Baseline ({baseline_avg:.2f}s)')
        plt.axhline(y=baseline_avg * 2, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='2x Baseline (Warning)')
        plt.axhline(y=baseline_avg * 3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='3x Baseline (Critical)')
    
    # Mark degradation events
    for event in degradation_events:
        color = 'orange' if event['severity'] == 'WARNING' else 'red'
        marker = 'o' if event['severity'] == 'WARNING' else 'X'
        plt.scatter(event['batch_num'], event['batch_time'], 
                   color=color, s=100, marker=marker, zorder=5,
                   label=f"{event['severity']} at batch {event['batch_num']}")
    
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('API Response Time (seconds)', fontsize=12)
    plt.title('Stress Test: API Response Time Over Batches', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/batch_time_over_batches.png', dpi=300)
    print(f"Saved: {output_folder}/batch_time_over_batches.png")
    plt.close()
    
    # 2. Batch Response Time vs Cumulative Transactions
    plt.figure(figsize=(14, 6))
    plt.plot(cumulative_transactions, batch_times, linewidth=1, alpha=0.7)
    
    if baseline_avg:
        plt.axhline(y=baseline_avg, color='green', linestyle='--', linewidth=2, label=f'Baseline ({baseline_avg:.2f}s)')
        plt.axhline(y=baseline_avg * 2, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='2x Baseline')
        plt.axhline(y=baseline_avg * 3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='3x Baseline')
    
    for event in degradation_events:
        color = 'orange' if event['severity'] == 'WARNING' else 'red'
        marker = 'o' if event['severity'] == 'WARNING' else 'X'
        txn_count = event['total_transactions']
        plt.scatter(txn_count, event['batch_time'], 
                   color=color, s=100, marker=marker, zorder=5)
        plt.annotate(f"{txn_count:,} txns", 
                    xy=(txn_count, event['batch_time']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold')
    
    plt.xlabel('Cumulative Transactions Processed', fontsize=12)
    plt.ylabel('API Response Time (seconds)', fontsize=12)
    plt.title('Stress Test: API Response Time vs Total Transactions', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/batch_time_vs_transactions.png', dpi=300)
    print(f"Saved: {output_folder}/batch_time_vs_transactions.png")
    plt.close()
    
    # 3. Moving Average (Smoothed Trend)
    window = 20
    if len(batch_times) >= window:
        moving_avg = np.convolve(batch_times, np.ones(window)/window, mode='valid')
        moving_avg_batches = batch_numbers[window-1:]
        
        plt.figure(figsize=(14, 6))
        plt.plot(batch_numbers, batch_times, alpha=0.3, linewidth=0.5, label='Raw Batch Time')
        plt.plot(moving_avg_batches, moving_avg, linewidth=2, color='blue', label=f'{window}-Batch Moving Average')
        
        if baseline_avg:
            plt.axhline(y=baseline_avg, color='green', linestyle='--', linewidth=2, label=f'Baseline ({baseline_avg:.2f}s)')
        
        for event in degradation_events:
            color = 'orange' if event['severity'] == 'WARNING' else 'red'
            plt.axvline(x=event['batch_num'], color=color, linestyle=':', alpha=0.5)
        
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel('API Response Time (seconds)', fontsize=12)
        plt.title('Stress Test: Smoothed Performance Trend', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_folder}/smoothed_trend.png', dpi=300)
        print(f"Saved: {output_folder}/smoothed_trend.png")
        plt.close()
    
    # 4. Degradation Timeline
    if degradation_events:
        plt.figure(figsize=(12, 6))
        
        warning_events = [e for e in degradation_events if e['severity'] == 'WARNING']
        critical_events = [e for e in degradation_events if e['severity'] == 'CRITICAL']
        
        if warning_events:
            warning_batches = [e['batch_num'] for e in warning_events]
            warning_factors = [e['degradation_factor'] for e in warning_events]
            plt.scatter(warning_batches, warning_factors, color='orange', s=100, marker='o', label='WARNING', zorder=5)
        
        if critical_events:
            critical_batches = [e['batch_num'] for e in critical_events]
            critical_factors = [e['degradation_factor'] for e in critical_events]
            plt.scatter(critical_batches, critical_factors, color='red', s=150, marker='X', label='CRITICAL', zorder=5)
        
        plt.axhline(y=2.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='2x Threshold')
        plt.axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='3x Threshold')
        
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel('Degradation Factor (vs Baseline)', fontsize=12)
        plt.title('Stress Test: Performance Degradation Events', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_folder}/degradation_timeline.png', dpi=300)
        print(f"Saved: {output_folder}/degradation_timeline.png")
        plt.close()
    
    # 5. Summary Statistics Box
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    summary_text = f"""
STRESS TEST SUMMARY

Total Batches: {len(batch_numbers):,}
Total Transactions: {total_transactions:,}
Total Time: {metrics['total_elapsed_time']:.2f}s ({metrics['total_elapsed_time']/60:.2f} min)

Performance Metrics:
  Baseline Average: {baseline_avg:.2f}s
  Overall Average: {metrics['avg_batch_time']:.2f}s
  Min Batch Time: {min(batch_times):.2f}s
  Max Batch Time: {max(batch_times):.2f}s
  
Degradation Events:
  WARNING (2x): {len([e for e in degradation_events if e['severity'] == 'WARNING'])}
  CRITICAL (3x): {len([e for e in degradation_events if e['severity'] == 'CRITICAL'])}
  
First Degradation: {degradation_events[0]['batch_num'] if degradation_events else 'None'}
  At Transaction: {degradation_events[0]['total_transactions']:,} if degradation_events else 'N/A'}
  
Errors: {len(metrics['errors'])}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/summary_stats.png', dpi=300)
    print(f"Saved: {output_folder}/summary_stats.png")
    plt.close()
    
    print(f"\n✅ All graphs saved to {output_folder}/")


def main():
    """
    Usage: python graph_stress_test.py
    """
    import sys
    
    if len(sys.argv) > 1:
        metrics_path = sys.argv[1]
    else:
        metrics_path = 'stress_test_output/stress_metrics.json'
    
    print(f"Loading metrics from: {metrics_path}")
    plot_stress_test_results(metrics_path)


if __name__ == "__main__":
    main()
