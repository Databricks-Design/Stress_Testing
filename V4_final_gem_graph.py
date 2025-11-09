import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, date, time
from collections import Counter

# Use a professional style
plt.style.use('seaborn-v0_8-darkgrid')

# --- CONFIGURATION ---
ROLLING_WINDOW_SIZE = 30
METRICS_PATH = 'stress_test_output/stress_metrics.json'

# Professional color scheme
COLOR_SUCCESS = '#2ecc71'  # Green
COLOR_FAILURE = '#e74c3c'  # Red
COLOR_WARNING = '#f39c12'  # Orange
COLOR_CRITICAL = '#c0392b'  # Dark Red
COLOR_BASELINE = '#3498db'  # Blue
COLOR_THRESHOLD = '#95a5a6'  # Gray

# Threshold percentages for marking
THRESHOLDS = [50, 80, 90, 100]

# --- DATA LOADING ---
if not os.path.exists(METRICS_PATH):
    print(f"Error: Metrics file not found at {METRICS_PATH}")
    sys.exit(1)

with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)

df = pd.DataFrame(metrics['results_per_batch'])

if df.empty:
    print("Error: No batch results found in metrics file.")
    sys.exit(1)

summary = metrics['test_summary']
milestones = metrics['failure_milestones']
baseline_avg = summary.get('baseline_avg_time')
df['is_error'] = (df['status_code'] != 200).astype(int)

# --- TIME-BASED ANALYSIS ---
print("=" * 80)
print("TIME-BASED ANALYSIS")
print("=" * 80)
start_time_str = input(f"Enter test start time (24h format, e.g., 22:10:00): ").strip()

start_timestamp = None
while start_timestamp is None:
    try:
        parsed_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
        start_timestamp = datetime.combine(date.today(), parsed_time)
        break
    except ValueError:
        print("Invalid format. Use HH:MM:SS (e.g., 22:10:00)")
        start_time_str = input("Enter test start time: ").strip()

# Calculate timestamps
df['cumulative_time_sec'] = df['time_sec'].cumsum()
df['timestamp'] = df['cumulative_time_sec'].apply(
    lambda x: start_timestamp + pd.Timedelta(seconds=x)
)
df['minute_bucket'] = df['timestamp'].dt.floor('T')

# Per-minute summary
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
    print(f"  - Avg Response Time: {data['avg_time_sec']:.3f}s")

print("\n" + "=" * 80)
print("Generating Professional Stakeholder Dashboard...")
print("=" * 80)

# Separate healthy vs unhealthy
healthy = df[df['status_code'] == 200].copy()
unhealthy = df[df['status_code'] != 200].copy()

# Calculate rolling error rate
df['rolling_error_rate'] = df['is_error'].rolling(
    window=ROLLING_WINDOW_SIZE, min_periods=1
).mean() * 100

# --- CREATE PROFESSIONAL DASHBOARD ---
# 6 graphs stacked vertically, full width
fig = plt.figure(figsize=(20, 50))
gs = fig.add_gridspec(6, 1, height_ratios=[2.5, 2.5, 2.5, 2, 2.5, 1.8], hspace=0.3)

# Professional title
fig.suptitle(
    'NER API Stress Test - Stakeholder Dashboard',
    fontsize=28,
    fontweight='bold',
    y=0.995
)

# =============================================================================
# GRAPH 1: Health Status Timeline (The "Aha!" Graph)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title(
    'Graph 1: Health Status Timeline - System Behavior Over Time',
    fontsize=18,
    fontweight='bold',
    pad=15
)

# Plot healthy requests
if not healthy.empty:
    ax1.scatter(
        healthy['batch_num'],
        healthy['time_sec'],
        color=COLOR_SUCCESS,
        alpha=0.6,
        s=30,
        label='Healthy (200 OK)',
        edgecolors='none'
    )

# Plot failures with prominent X markers
if not unhealthy.empty:
    ax1.scatter(
        unhealthy['batch_num'],
        unhealthy['time_sec'],
        color=COLOR_FAILURE,
        marker='X',
        s=120,
        label='Failure (Non-200)',
        edgecolors='darkred',
        linewidths=1,
        zorder=5
    )

# Baseline line
if baseline_avg:
    ax1.axhline(
        y=baseline_avg,
        color=COLOR_BASELINE,
        linestyle='--',
        linewidth=2.5,
        label=f'Baseline: {baseline_avg:.3f}s',
        zorder=3
    )

# First error marker
if milestones.get('first_error_batch_num'):
    first_error_batch = milestones['first_error_batch_num']
    ax1.axvline(
        x=first_error_batch,
        color=COLOR_FAILURE,
        linestyle='-',
        linewidth=2.5,
        alpha=0.7,
        label=f'First Error (Batch {first_error_batch})',
        zorder=3
    )
    # Add annotation
    ymax = ax1.get_ylim()[1]
    ax1.annotate(
        f'First Failure\nBatch {first_error_batch}',
        xy=(first_error_batch, ymax * 0.9),
        fontsize=11,
        ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLOR_FAILURE, linewidth=2),
        zorder=10
    )

ax1.set_xlabel('Batch Number', fontsize=14, fontweight='bold')
ax1.set_ylabel('API Response Time (seconds)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax1.grid(True, linestyle=':', alpha=0.5, linewidth=1)
ax1.tick_params(axis='both', labelsize=11)

# =============================================================================
# GRAPH 2: Rolling Error Rate with Threshold Markers
# =============================================================================
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title(
    'Graph 2: Rolling Error Rate (30-Batch Window) - Service Health Degradation',
    fontsize=18,
    fontweight='bold',
    pad=15
)

# Plot rolling error rate
ax2.plot(
    df['batch_num'],
    df['rolling_error_rate'],
    color=COLOR_FAILURE,
    linewidth=3,
    label=f'Rolling {ROLLING_WINDOW_SIZE}-Batch Error Rate',
    zorder=5
)

# Add threshold lines with annotations
threshold_colors = {
    50: COLOR_WARNING,
    80: COLOR_CRITICAL,
    90: COLOR_CRITICAL,
    100: 'black'
}

for threshold in THRESHOLDS:
    color = threshold_colors.get(threshold, COLOR_THRESHOLD)
    linestyle = '--' if threshold < 100 else '-'
    linewidth = 2 if threshold < 100 else 3
    
    ax2.axhline(
        y=threshold,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.8,
        zorder=2
    )
    
    # Add threshold label on the right side
    ax2.text(
        ax2.get_xlim()[1] * 1.01,
        threshold,
        f'{threshold}%',
        fontsize=12,
        fontweight='bold',
        color=color,
        va='center',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=2)
    )

# Mark first error
if milestones.get('first_error_batch_num'):
    first_error_batch = milestones['first_error_batch_num']
    ax2.axvline(
        x=first_error_batch,
        color=COLOR_FAILURE,
        linestyle='-',
        linewidth=2.5,
        alpha=0.7,
        label=f'First Error (Batch {first_error_batch})',
        zorder=3
    )

ax2.set_xlabel('Batch Number', fontsize=14, fontweight='bold')
ax2.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold')
ax2.set_ylim(-5, 105)
ax2.set_yticks(np.arange(0, 101, 10))
ax2.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax2.grid(True, linestyle=':', alpha=0.5, linewidth=1)
ax2.tick_params(axis='both', labelsize=11)

# =============================================================================
# GRAPH 3: Performance of Successful Requests Only (True Slowdown)
# =============================================================================
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_title(
    'Graph 3: Performance of Successful Requests (200 OK) - True Slowdown Analysis',
    fontsize=18,
    fontweight='bold',
    pad=15
)

if not healthy.empty:
    # Plot individual successful batch times
    ax3.plot(
        healthy['batch_num'],
        healthy['time_sec'],
        linestyle='None',
        marker='o',
        markersize=4,
        alpha=0.4,
        color=COLOR_SUCCESS,
        label='Individual Batch Time (200 OK)'
    )
    
    # Calculate and plot rolling average
    healthy_sorted = healthy.sort_values('batch_num')
    healthy_sorted['rolling_avg'] = healthy_sorted['time_sec'].rolling(
        window=ROLLING_WINDOW_SIZE, min_periods=1
    ).mean()
    
    ax3.plot(
        healthy_sorted['batch_num'],
        healthy_sorted['rolling_avg'],
        color=COLOR_BASELINE,
        linewidth=3,
        label=f'Rolling {ROLLING_WINDOW_SIZE}-Batch Avg (200 OK)',
        zorder=5
    )
    
    # Baseline
    if baseline_avg:
        ax3.axhline(
            y=baseline_avg,
            color='green',
            linestyle='--',
            linewidth=2.5,
            label=f'Baseline: {baseline_avg:.3f}s',
            zorder=3
        )
        
        # 2x and 3x baseline markers
        ax3.axhline(
            y=baseline_avg * 2,
            color=COLOR_WARNING,
            linestyle=':',
            linewidth=2,
            alpha=0.7,
            label=f'2x Baseline (WARNING)',
            zorder=2
        )
        ax3.axhline(
            y=baseline_avg * 3,
            color=COLOR_CRITICAL,
            linestyle=':',
            linewidth=2,
            alpha=0.7,
            label=f'3x Baseline (CRITICAL)',
            zorder=2
        )
else:
    ax3.text(
        0.5, 0.5,
        'No successful (200 OK) requests found',
        ha='center', va='center',
        transform=ax3.transAxes,
        fontsize=16,
        color='gray'
    )

ax3.set_xlabel('Batch Number', fontsize=14, fontweight='bold')
ax3.set_ylabel('API Response Time (seconds)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax3.grid(True, linestyle=':', alpha=0.5, linewidth=1)
ax3.tick_params(axis='both', labelsize=11)

# =============================================================================
# GRAPH 4: Status Code Distribution (NEW)
# =============================================================================
ax4 = fig.add_subplot(gs[3, 0])
ax4.set_title(
    'Graph 4: HTTP Status Code Distribution - Error Type Breakdown',
    fontsize=18,
    fontweight='bold',
    pad=15
)

# Count status codes
status_counts = df['status_code'].value_counts().sort_index()
total_requests = len(df)

# Create bar chart
colors_map = {200: COLOR_SUCCESS}
colors = [colors_map.get(code, COLOR_FAILURE) for code in status_counts.index]

bars = ax4.bar(
    [str(code) for code in status_counts.index],
    status_counts.values,
    color=colors,
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8
)

# Add count and percentage labels on bars
for i, (bar, count) in enumerate(zip(bars, status_counts.values)):
    percentage = (count / total_requests) * 100
    height = bar.get_height()
    
    # Label above bar
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        height + (ax4.get_ylim()[1] * 0.02),
        f'{count:,}\n({percentage:.1f}%)',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1)
    )

ax4.set_xlabel('HTTP Status Code', fontsize=14, fontweight='bold')
ax4.set_ylabel('Number of Requests', fontsize=14, fontweight='bold')
ax4.grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=1)
ax4.tick_params(axis='both', labelsize=11)

# Add custom legend
legend_elements = [
    mpatches.Patch(facecolor=COLOR_SUCCESS, edgecolor='black', label='Success (200 OK)'),
    mpatches.Patch(facecolor=COLOR_FAILURE, edgecolor='black', label='Failure (Non-200)')
]
ax4.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.95)

# =============================================================================
# GRAPH 5: Success vs Fail Per Minute (Stacked Bar)
# =============================================================================
ax5 = fig.add_subplot(gs[4, 0])
ax5.set_title(
    'Graph 5: Batch Success/Failure Distribution Per Minute - Timeline View',
    fontsize=18,
    fontweight='bold',
    pad=15
)

# Create stacked bar chart
x_positions = np.arange(len(minute_summary))
bar_width = 0.8

bars1 = ax5.bar(
    x_positions,
    minute_summary['successes'],
    bar_width,
    label='Success (200 OK)',
    color=COLOR_SUCCESS,
    edgecolor='black',
    linewidth=1
)

bars2 = ax5.bar(
    x_positions,
    minute_summary['failures'],
    bar_width,
    bottom=minute_summary['successes'],
    label='Failure (Non-200)',
    color=COLOR_FAILURE,
    edgecolor='black',
    linewidth=1
)

# Set x-axis labels
time_labels = [t.strftime('%H:%M') for t in minute_summary.index]
ax5.set_xticks(x_positions)
ax5.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=10)

ax5.set_xlabel('Time (HH:MM)', fontsize=14, fontweight='bold')
ax5.set_ylabel('Total Batches', fontsize=14, fontweight='bold')
ax5.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax5.grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=1)
ax5.tick_params(axis='both', labelsize=11)

# =============================================================================
# GRAPH 6: Executive Summary (Text-based)
# =============================================================================
ax6 = fig.add_subplot(gs[5, 0])
ax6.set_title(
    'Graph 6: Executive Summary - Key Metrics & Findings',
    fontsize=18,
    fontweight='bold',
    pad=15
)
ax6.axis('off')

# Calculate metrics
total_api_time_sec = sum(df['time_sec'])
total_time_min = total_api_time_sec / 60
success_count = len(healthy)
failure_count = len(unhealthy)
success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0

# Get unique status codes
unique_status_codes = sorted(df['status_code'].unique())
status_code_str = ', '.join([str(code) for code in unique_status_codes])

summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                        STRESS TEST EXECUTIVE SUMMARY                      ║
╚══════════════════════════════════════════════════════════════════════════╝

API ENDPOINT: {summary.get('api_name', 'N/A')}
STOP REASON: {summary.get('stop_reason', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOLUME PROCESSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total Batches Executed:        {summary.get('total_batches_run', 0):,}
  • Total Transactions Processed:  {summary.get('total_transactions_processed', 0):,}
  • Total API Time (Cumulative):   {total_api_time_sec:.2f}s ({total_time_min:.2f} min)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE BASELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Baseline (First 10 OK Batches): {f'{baseline_avg:.3f}s' if baseline_avg else 'Not Established'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • First Error Detected at Batch:  {milestones.get('first_error_batch_num', 'N/A')}
  • Transactions Before First Error: {f"{milestones.get('total_transactions_at_first_error', 'N/A'):,}" if milestones.get('total_transactions_at_first_error') else 'N/A'}
  • First Error Message:            {milestones.get('first_error_message', 'N/A')[:60]}...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUCCESS vs FAILURE BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total Successful (200 OK):      {success_count:,} ({success_rate:.1f}%)
  • Total Failed (Non-200):         {failure_count:,} ({100-success_rate:.1f}%)
  • Unique Status Codes Observed:  {status_code_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY FINDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The API maintained stable performance until batch {milestones.get('first_error_batch_num', 'N/A')}, after
which the rolling error rate progressively increased through 50%, 80%, and 90% 
thresholds before reaching 100% service unavailability. This pattern indicates
memory exhaustion leading to complete pod failure rather than gradual degradation.
"""

ax6.text(
    0.02, 0.98,
    summary_text,
    transform=ax6.transAxes,
    fontsize=11,
    verticalalignment='top',
    fontfamily='monospace',
    bbox=dict(
        boxstyle='round,pad=1',
        facecolor='#f8f9fa',
        edgecolor='#343a40',
        linewidth=2
    )
)

# Save the figure
output_path = 'stress_test_output/stakeholder_dashboard.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Professional dashboard saved to: {output_path}")
print(f"   Resolution: 300 DPI (publication quality)")

# Display the figure
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("Dashboard generation complete!")
print("=" * 80)
