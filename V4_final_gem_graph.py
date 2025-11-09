import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, date, time

# --- 1. CONFIGURATION ---

# This MUST match the ROLLING_WINDOW_SIZE in your stress_test.py script
ROLLING_WINDOW_SIZE = 30

# Set the path to your metrics file
METRICS_PATH = 'stress_test_output/stress_metrics.json'

# --- 2. DATA LOADING & TIME-BASED ANALYSIS ---

if not os.path.exists(METRICS_PATH):
    print(f"Error: Metrics file not found at {METRICS_PATH}")
    print("Please make sure the file is in the correct location and re-run.")
else:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)

    # Load data into a pandas DataFrame for easy analysis
    df = pd.DataFrame(metrics['results_per_batch'])
    
    if df.empty:
        print("Error: No batch results found in metrics file.")
    else:
        summary = metrics['test_summary']
        milestones = metrics['failure_milestones']
        baseline_avg = summary.get('baseline_avg')
        df['is_error'] = (df['status_code'] != 200).astype(int)

        # --- Ask for Start Time and Print Minute-by-Minute Summary ---
        print("--- TIME-BASED ANALYSIS ---")
        start_time_str = input(f"Enter the test start time (24h format, e.g., 20:10:00 for {datetime.now().strftime('%H:%M:%S')}): ")
        
        start_timestamp = None
        while start_timestamp is None:
            try:
                parsed_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
                start_timestamp = datetime.combine(date.today(), parsed_time)
            except ValueError:
                print("Invalid time format. Please use HH:MM:SS (e.g., 20:10:00).")
                start_time_str = input("Enter the test start time: ")

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
            print(f"  - Total Batches: {int(data['total_batches'])}")
            print(f"  - Successes (200 OK): {int(data['successes'])}")
            print(f"  - Failures (Non-200): {int(data['failures'])}")
            print(f"  - Avg. Response Time: {data['avg_time_sec']:.3f}s")
        print("\n" + "="*80)
        print("Generating Stakeholder Dashboard... (This may take a moment)")
        
        # Separate healthy vs. unhealthy results
        healthy = df[df['status_code'] == 200].copy()
        unhealthy = df[df['status_code'] != 200].copy()

        # --- 3. PLOTTING THE 5-GRAPH DASHBOARD ---
        
        # Create a single figure with 5 subplots stacked vertically
        # We make it very tall to ensure each graph is clear
        fig, (ax1, ax2, ax3, ax5, ax4) = plt.subplots(
            5, 1, 
            figsize=(18, 45), # Full width (18) and very tall (45)
            gridspec_kw={'height_ratios': [2, 2, 2, 2, 1.5]} # Give last plot less space
        )
        
        # Add a professional main title
        fig.suptitle('Stress Test Investigation Dashboard', fontsize=24, fontweight='bold', y=1.0)
        
        # ---
        # Graph 1: The "Aha!" Graph (Health Status Timeline)
        # ---
        ax1.set_title('Graph 1: Health Status Timeline (The "Aha!" Graph)', fontsize=18, fontweight='bold', loc='left')
        ax1.scatter(
            healthy['batch_num'], 
            healthy['time_sec'], 
            color='green', 
            alpha=0.5, 
            s=15, 
            label='Healthy (200 OK)'
        )
        ax1.scatter(
            unhealthy['batch_num'], 
            unhealthy['time_sec'], 
            color='red', 
            marker='x', 
            s=50, 
            label=f'Failure (Non-200)'
        )
        if baseline_avg:
            ax1.axhline(y=baseline_avg, color='green', linestyle='--', label=f'Baseline ({baseline_avg:.3f}s)')
        if milestones.get('first_error_batch_num'):
            first_error_batch = milestones['first_error_batch_num']
            ax1.axvline(
                x=first_error_batch, 
                color='red', 
                linestyle='--', 
                label=f"First Error at Batch {first_error_batch}"
            )
        ax1.set_xlabel('Batch Number', fontsize=12)
        ax1.set_ylabel('API Response Time (seconds)', fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)

        # ---
        # Graph 2: The "Health Dashboard" (Rolling Error Rate)
        # ---
        ax2.set_title('Graph 2: "Health Dashboard" (Rolling Error Rate %)', fontsize=18, fontweight='bold', loc='left')
        df['rolling_error_rate'] = df['is_error'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean() * 100
        ax2.plot(df['batch_num'], df['rolling_error_rate'], color='red', linewidth=2, label=f'Rolling {ROLLING_WINDOW_SIZE}-Batch Error Rate')
        if milestones.get('first_error_batch_num'):
            first_error_batch = milestones['first_error_batch_num']
            ax2.axvline(
                x=first_error_batch, 
                color='red', 
                linestyle='--', 
                label=f"First Error at Batch {first_error_batch}"
            )
        ax2.set_xlabel('Batch Number', fontsize=12)
        ax2.set_ylabel('Failure Rate (%)', fontsize=12)
        ax2.set_yticks(np.arange(0, 101, 10))
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)

        # ---
        # Graph 3: The "True Slowdown" (Successful Requests Only)
        # ---
        ax3.set_title('Graph 3: "True Slowdown" (Performance of Successful 200 OK Requests Only)', fontsize=18, fontweight='bold', loc='left')
        if not healthy.empty:
            ax3.plot(
                healthy['batch_num'], 
                healthy['time_sec'], 
                linestyle='None', 
                marker='.', 
                alpha=0.5, 
                color='green',
                label='Successful (200 OK) Batch Time'
            )
            healthy.loc[:, 'rolling_avg_time'] = healthy['time_sec'].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
            ax3.plot(
                healthy['batch_num'], 
                healthy['rolling_avg_time'], 
                color='blue', 
                linewidth=2,
                label=f'Rolling {ROLLING_WINDOW_SIZE}-Batch Avg (Healthy)'
            )
            if baseline_avg:
                ax3.axhline(y=baseline_avg, color='green', linestyle='--', label=f'Baseline ({baseline_avg:.3f}s)')
        else:
            ax3.text(0.5, 0.5, "No successful (200 OK) requests were found.", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=14, color='gray')
        ax3.set_xlabel('Batch Number', fontsize=12)
        ax3.set_ylabel('API Response Time (seconds)', fontsize=12)
        ax3.legend()
        ax3.grid(True, linestyle=':', alpha=0.6)

        # ---
        # Graph 4: Success vs. Fail (Per Minute)
        # ---
        ax5.set_title('Graph 4: Success vs. Fail (Per Minute)', fontsize=18, fontweight='bold', loc='left')
        plot_data = minute_summary[['successes', 'failures']]
        plot_data.plot(kind='bar', stacked=True, color=['green', 'red'], ax=ax5, width=0.8, alpha=0.8)
        ax5.set_xticklabels([t.strftime('%H:%M') for t in plot_data.index], rotation=45, ha='right')
        ax5.set_xlabel('Time (Minute)', fontsize=12)
        ax5.set_ylabel('Total Batches (Success + Fail)', fontsize=12)
        ax5.legend(['Successes (200 OK)', 'Failures (Non-200)'])
        ax5.grid(True, linestyle=':', alpha=0.6, axis='y')

        # ---
        # Graph 5: The "Executive Summary"
        # ---
        ax4.set_title('Graph 5: Executive Summary', fontsize=18, fontweight='bold', loc='left')
        ax4.axis('off') # This is a text-only plot

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
        
        ax4.text(0.01, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=13, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=1))

        # --- Show the final dashboard ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make room for suptitle
        plt.show()
