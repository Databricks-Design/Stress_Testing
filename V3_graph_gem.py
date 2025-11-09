import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, date, time

# --- CONFIGURATION ---
# This MUST match the ROLLING_WINDOW_SIZE in your stress_test.py script
ROLLING_WINDOW_SIZE = 20
# ---------------------

def plot_stress_test_results(metrics_path: str, output_folder: str = 'stress_test_graphs'):
    """
    Create comprehensive graphs from the investigator script's metrics.
    Includes a new time-based analysis.
    """
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Please run the stress_test.py script first.")
        return

    # Create the output folder relative to the metrics file
    base_output_folder = os.path.dirname(metrics_path)
    if base_output_folder == "":
        base_output_folder = "." # Handle running in the same directory
    graph_output_folder = os.path.join(base_output_folder, output_folder)
    os.makedirs(graph_output_folder, exist_ok=True)
    print(f"Saving graphs to: {graph_output_folder}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Load data into a pandas DataFrame for easy analysis
    df = pd.DataFrame(metrics['results_per_batch'])
    if df.empty:
        print("Error: No batch results found in metrics file.")
        return
        
    summary = metrics['test_summary']
    milestones = metrics['failure_milestones']
    baseline_avg = summary.get('baseline_avg')
    df['is_error'] = (df['status_code'] != 200).astype(int)

    # ---
    # NEW: Time-Based Analysis (Your Request)
    # ---
    print("\n--- TIME-BASED ANALYSIS ---")
    start_time_str = input("Enter the test start time (24h format, e.g., 20:10:00 for 8:10 PM): ")
    
    try:
        parsed_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
    except ValueError:
        print("Invalid time format. Using 00:00:00 as default.")
        parsed_time = time(0, 0, 0)

    start_timestamp = datetime.combine(date.today(), parsed_time)
    
    # Calculate timestamps for every batch
    df['cumulative_time_sec'] = df['time_sec'].cumsum()
    df['timestamp'] = df['cumulative_time_sec'].apply(lambda x: start_timestamp + pd.Timedelta(seconds=x))
    df['minute_bucket'] = df['timestamp'].dt.floor('T') # 'T' = Minute frequency

    # Create the per-minute summary
    minute_summary = df.groupby('minute_bucket').agg(
        total_batches=pd.NamedAgg(column='batch_num', aggfunc='count'),
        successes=pd.NamedAgg(column='is_error', aggfunc=lambda x: (x == 0).sum()),
        failures=pd.NamedAgg(column='is_error', aggfunc='sum'),
        avg_time_sec=pd.NamedAgg(column='time_sec', aggfunc='mean')
    )

    print("\n--- MINUTE-BY-MINUTE SUMMARY ---")
    for minute, data in minute_summary.iterrows():
        print(f"\nTime: {minute.strftime('%Y-%m-%d %H:%M')}")
        print(f"  - Total Batches: {data['total_batches']}")
        print(f"  - Successes (200 OK): {data['successes']}")
        print(f"  - Failures (Non-200): {data['failures']}")
        print(f"  - Avg. Response Time: {data['avg_time_sec']:.3f}s")
    
    # Separate healthy vs. unhealthy results
    healthy = df[df['status_code'] == 200].copy() # Use .copy() to avoid SettingWithCopyWarning
    unhealthy = df[df['status_code'] != 200].copy()

    # ---
    # Graph 1: The "Aha!" Graph (Health Status Timeline)
    # ---
    print("\nGenerating Graph 1: Health Status Timeline...")
    plt.figure(figsize=(15, 7))
    
    plt.scatter(
        healthy['batch_num'], 
        healthy['time_sec'], 
        color='green', 
        alpha=0.5, 
        s=10, 
        label='Healthy (200 OK)'
    )
    plt.scatter(
        unhealthy['batch_num'], 
        unhealthy['time_sec'], 
        color='red', 
        marker='x', 
        s=40, 
        label=f'Failure (Non-200)'
    )
    
    if baseline_avg:
        plt.axhline(y=baseline_avg, color='green', linestyle='--', label=f'Baseline ({baseline_avg:.3f}s)')

    if milestones.get('first_error_batch_num'):
        first_error_batch = milestones['first_error_batch_num']
        plt.axvline(
            x=first_error_batch, 
            color='red', 
            linestyle='--', 
            label=f"First Error at Batch {first_error_batch}"
        )

    plt.title('Graph 1: Health Status Timeline (The "Aha!" Graph)', fontsize=16, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('API Response Time (seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_folder, '1_health_status_timeline.png'), dpi=120)
    plt.close()

    # ---
    # Graph 2: The "Health Dashboard" (Rolling Error Rate)
    # ---
    print("Generating Graph 2: Rolling Error Rate...")
    if not df.empty:
        df['rolling_error_rate'] = df['is_error'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean() * 100
        
        plt.figure(figsize=(15, 7))
        plt.plot(df['batch_num'], df['rolling_error_rate'], color='red', label=f'Rolling {ROLLING_WINDOW_SIZE}-Batch Error Rate')
        
        if milestones.get('first_error_batch_num'):
            first_error_batch = milestones['first_error_batch_num']
            plt.axvline(
                x=first_error_batch, 
                color='red', 
                linestyle='--', 
                label=f"First Error at Batch {first_error_batch}"
            )

        plt.title('Graph 2: "Health Dashboard" (Rolling Error Rate %)', fontsize=16, fontweight='bold')
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel('Failure Rate (%)', fontsize=12)
        plt.yticks(np.arange(0, 101, 10)) # Y-axis from 0% to 100%
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_output_folder, '2_rolling_error_rate.png'), dpi=120)
        plt.close()

    # ---
    # Graph 3: The "True Slowdown" (Successful Requests Only)
    # ---
    print("Generating Graph 3: Healthy Request Performance...")
    if not healthy.empty:
        plt.figure(figsize=(15, 7))
        plt.plot(
            healthy['batch_num'], 
            healthy['time_sec'], 
            linestyle='None', 
            marker='.', 
            alpha=0.5, 
            label='Successful (200 OK) Batch Time'
        )
        
        healthy.loc[:, 'rolling_avg_time'] = healthy['time_sec'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
        plt.plot(
            healthy['batch_num'], 
            healthy['rolling_avg_time'], 
            color='blue', 
            label=f'Rolling {ROLLING_WINDOW_SIZE}-Batch Avg (Healthy)'
        )
        
        if baseline_avg:
            plt.axhline(y=baseline_avg, color='green', linestyle='--', label=f'Baseline ({baseline_avg:.3f}s)')

        plt.title('Graph 3: "True Slowdown" (Performance of Successful 200 OK Requests Only)', fontsize=16, fontweight='bold')
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel('API Response Time (seconds)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_output_folder, '3_healthy_request_performance.png'), dpi=120)
        plt.close()
    else:
        print("Skipping Graph 3: No successful (200 OK) requests were found.")
        
    # ---
    # Graph 4: The "Executive Summary"
    # ---
    print("Generating Graph 4: Executive Summary...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    total_api_time_sec = sum(df['time_sec'])
    total_time_min = total_api_time_sec / 60
    
    summary_text = f"""
    STRESS TEST: EXECUTIVE SUMMARY
    ----------------------------------------------------------
    API Endpoint: {summary.get('api_name', 'N/A')}
    Stop Reason: {summary.get('stop_reason', 'N/A')}
    
    Total Batches Run: {summary.get('total_batches_run', 0):,}
    Total Transactions Processed: {summary.get('total_transactions_processed', 0):,}
    Total API Time (Sum): {total_api_time_sec:.2f}s ({total_time_min:.2f} min)
    
    PERFORMANCE
    ----------------------------------------------------------
    Baseline (First 10 OK): {f'{baseline_avg:.3f}s' if baseline_avg else 'Not Established'}
    
    FAILURE ANALYSIS
    ----------------------------------------------------------
    First Error Batch: {milestones.get('first_error_batch_num') or 'N/A'}
    Transactions at First Error: {milestones.get('total_transactions_at_first_error', 'N/A'):,}
    First Error Message: {milestones.get('first_error_message') or 'N/A'}
    
    Total 200 OK Requests: {len(healthy):,}
    Total Failed Requests: {len(unhealthy):,}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=1))

    plt.title('Graph 4: Executive Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_folder, '4_executive_summary.png'), dpi=120)
    plt.close()

    # ---
    # NEW: Graph 5: Success vs. Fail (Per Minute)
    # ---
    print("Generating Graph 5: Success vs. Fail (Per Minute)...")
    
    # We need to re-format the data for a stacked bar chart
    plot_data = minute_summary[['successes', 'failures']]
    
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data.plot(kind='bar', stacked=True, color=['green', 'red'], ax=ax, width=0.8)

    # Format the X-axis labels to show HH:MM
    ax.set_xticklabels([t.strftime('%H:%M') for t in plot_data.index], rotation=45, ha='right')

    plt.title('Graph 5: Success vs. Fail (Per Minute)', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Minute)', fontsize=12)
    plt.ylabel('Total Batches (Success + Fail)', fontsize=12)
    plt.legend(['Successes (200 OK)', 'Failures (Non-200)'])
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_folder, '5_success_vs_fail_per_minute.png'), dpi=120)
    plt.close()


    print(f"\n✅ All 5 graphs saved to {graph_output_folder}/")


def main():
    if len(sys.argv) < 2:
        print("--- API Stress Test Graph Generator ---")
        print("\nUsage: python graph_stress_test.py <path_to_stress_metrics.json>")
        print(f"Example: python graph_stress_test.py stress_test_output/stress_metrics.json")
        sys.exit(1)
        
    metrics_path = sys.argv[1]
    # Default output folder is 'graphs' inside the metrics folder
    output_folder = "graphs" 
    
    plot_stress_test_results(metrics_path, output_folder)

if __name__ == "__main__":
    main()
