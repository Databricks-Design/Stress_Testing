import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, date, time
from collections import Counter

# Professional corporate styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3

# Modern corporate color palette
COLORS = {
    'success': '#10b981',      # Modern green
    'failure': '#ef4444',      # Modern red
    'warning': '#f59e0b',      # Amber
    'critical': '#dc2626',     # Deep red
    'info': '#3b82f6',         # Blue
    'baseline': '#8b5cf6',     # Purple
    'bg_light': '#f9fafb',     # Light gray background
    'text_dark': '#1f2937',    # Dark text
    'accent': '#06b6d4'        # Cyan accent
}

# --- CONFIGURATION ---
ROLLING_WINDOW_SIZE = 30
METRICS_PATH = 'stress_test_output/stress_metrics.json'

print("="*80)
print("PROFESSIONAL STAKEHOLDER DASHBOARD GENERATOR")
print("="*80)

# --- LOAD DATA ---
if not os.path.exists(METRICS_PATH):
    print(f"❌ Error: Metrics file not found at {METRICS_PATH}")
    sys.exit(1)

with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)

df = pd.DataFrame(metrics['results_per_batch'])
if df.empty:
    print("❌ Error: No batch results found")
    sys.exit(1)

summary = metrics['test_summary']
milestones = metrics['failure_milestones']
baseline_avg = summary.get('baseline_avg_time')
df['is_error'] = (df['status_code'] != 200).astype(int)

# Calculate rolling error rate
df['rolling_error_rate'] = df['is_error'].rolling(
    window=ROLLING_WINDOW_SIZE, min_periods=1
).mean() * 100

# --- TIME INPUT ---
print("\n📅 TIME-BASED ANALYSIS")
start_time_str = input("Enter test start time (24h format, e.g., 22:10:00): ").strip()

start_timestamp = None
while start_timestamp is None:
    try:
        parsed_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
        start_timestamp = datetime.combine(date.today(), parsed_time)
        break
    except ValueError:
        print("❌ Invalid format. Use HH:MM:SS")
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

# --- IDENTIFY THRESHOLD CROSSING POINTS ---
thresholds_data = []
threshold_percentages = [50, 80, 90, 100]

for threshold in threshold_percentages:
    # Find first batch where rolling error rate >= threshold
    crossing_batches = df[df['rolling_error_rate'] >= threshold]
    if not crossing_batches.empty:
        first_cross = crossing_batches.iloc[0]
        batch_num = first_cross['batch_num']
        
        # Get the window
        window_start = max(1, batch_num - ROLLING_WINDOW_SIZE + 1)
        window_end = batch_num
        
        # Count failures and successes in this window
        window_df = df[(df['batch_num'] >= window_start) & (df['batch_num'] <= window_end)]
        failures = window_df['is_error'].sum()
        successes = len(window_df) - failures
        
        thresholds_data.append({
            'threshold': threshold,
            'batch_num': batch_num,
            'window_start': window_start,
            'window_end': window_end,
            'failures': int(failures),
            'successes': int(successes),
            'error_rate': first_cross['rolling_error_rate']
        })

healthy = df[df['status_code'] == 200].copy()
unhealthy = df[df['status_code'] != 200].copy()

# Calculate key metrics
total_batches = len(df)
first_error_batch = milestones.get('first_error_batch_num')
if first_error_batch:
    batches_until_failure = first_error_batch
    txns_until_failure = milestones.get('total_transactions_at_first_error', 0)
else:
    batches_until_failure = total_batches
    txns_until_failure = summary.get('total_transactions_processed', 0)

print("\n🎨 Generating professional dashboard...")

# --- CREATE FIGURE ---
fig = plt.figure(figsize=(24, 48), facecolor='white')
gs = fig.add_gridspec(6, 1, height_ratios=[2.8, 3.2, 2.8, 2.2, 2.8, 2], hspace=0.35)

# Main title with styling
fig.text(0.5, 0.985, 'NER API Stress Test Analysis', 
         ha='center', fontsize=32, fontweight='bold', 
         color=COLORS['text_dark'])
fig.text(0.5, 0.975, 'Comprehensive Performance & Reliability Report',
         ha='center', fontsize=16, color='#6b7280', style='italic')

# =============================================================================
# GRAPH 1: Health Status Timeline with Smart Annotations
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#fafafa')

# Title with underline
title_text = ax1.text(0.02, 1.08, 'Health Status Timeline', 
                      transform=ax1.transAxes, fontsize=20, 
                      fontweight='bold', color=COLORS['text_dark'])
subtitle = ax1.text(0.02, 1.03, 'Visual representation of API response health over test duration',
                   transform=ax1.transAxes, fontsize=11, color='#6b7280', style='italic')

# Plot with gradient-like effect for healthy requests
if not healthy.empty:
    scatter1 = ax1.scatter(healthy['batch_num'], healthy['time_sec'],
                          c=healthy['batch_num'], cmap='Greens',
                          alpha=0.6, s=25, edgecolors='none',
                          label='Healthy (200 OK)')

# Plot failures with emphasis
if not unhealthy.empty:
    ax1.scatter(unhealthy['batch_num'], unhealthy['time_sec'],
               color=COLORS['failure'], marker='X', s=150,
               edgecolors='darkred', linewidths=1.5,
               label='Failed Requests', zorder=10)

# Baseline with styled annotation
if baseline_avg:
    line = ax1.axhline(y=baseline_avg, color=COLORS['baseline'],
                       linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Baseline: {baseline_avg:.3f}s', zorder=5)
    
    # Add baseline info box
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=COLORS['baseline'], linewidth=2, alpha=0.95)
    ax1.text(0.98, 0.95, f'Baseline Performance\n{baseline_avg:.3f}s',
            transform=ax1.transAxes, fontsize=11, ha='right', va='top',
            bbox=bbox_props, color=COLORS['baseline'], fontweight='bold')

# First error with arrow annotation
if first_error_batch:
    ax1.axvline(x=first_error_batch, color=COLORS['failure'],
               linestyle='-', linewidth=2.5, alpha=0.7, zorder=4)
    
    # Smart annotation with arrow
    ypos = ax1.get_ylim()[1] * 0.85
    bbox_props = dict(boxstyle='round,pad=0.6', facecolor=COLORS['failure'],
                     edgecolor='darkred', linewidth=2.5, alpha=0.95)
    text = ax1.text(first_error_batch, ypos,
                   f'  FIRST FAILURE  \nBatch {first_error_batch:,}\n{txns_until_failure:,} transactions',
                   fontsize=10, ha='left', va='top', color='white',
                   bbox=bbox_props, fontweight='bold', zorder=15)
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='darkred')])
    
    # Arrow pointing to failure
    arrow = FancyArrowPatch((first_error_batch, ypos * 0.85),
                           (first_error_batch, ypos * 0.5),
                           arrowstyle='->', mutation_scale=25,
                           color=COLORS['failure'], linewidth=3, zorder=14)
    ax1.add_patch(arrow)

ax1.set_xlabel('Batch Number', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax1.set_ylabel('API Response Time (seconds)', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
ax1.grid(True, linestyle=':', alpha=0.4, linewidth=1, color='gray')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# =============================================================================
# GRAPH 2: Rolling Error Rate with Intelligent Threshold Annotations
# =============================================================================
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#fafafa')

# Title
ax2.text(0.02, 1.08, 'Rolling Error Rate Progression', 
         transform=ax2.transAxes, fontsize=20, 
         fontweight='bold', color=COLORS['text_dark'])
ax2.text(0.02, 1.03, f'30-batch rolling window showing service degradation pattern',
        transform=ax2.transAxes, fontsize=11, color='#6b7280', style='italic')

# Plot error rate with gradient fill
ax2.plot(df['batch_num'], df['rolling_error_rate'],
        color=COLORS['failure'], linewidth=3.5, label='Rolling Error Rate',
        zorder=5, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])

# Fill under the curve
ax2.fill_between(df['batch_num'], 0, df['rolling_error_rate'],
                alpha=0.2, color=COLORS['failure'])

# Add threshold lines and SMART ANNOTATIONS
threshold_colors = {50: COLORS['warning'], 80: COLORS['critical'], 
                   90: COLORS['critical'], 100: 'black'}

for thresh_data in thresholds_data:
    threshold = thresh_data['threshold']
    batch_num = thresh_data['batch_num']
    color = threshold_colors.get(threshold, 'gray')
    
    # Threshold line
    ax2.axhline(y=threshold, color=color, linestyle='--',
               linewidth=2, alpha=0.7, zorder=2)
    
    # Right-side percentage label
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=color, linewidth=2.5, alpha=1)
    ax2.text(1.01, threshold/100, f'{threshold}%',
            transform=ax2.get_yaxis_transform(),
            fontsize=12, fontweight='bold', color=color,
            ha='left', va='center', bbox=bbox_props)
    
    # INTELLIGENT ANNOTATION - Show batch info and counts
    window_str = f"Batches {thresh_data['window_start']:,}-{thresh_data['window_end']:,}"
    counts_str = f"{thresh_data['failures']} fails, {thresh_data['successes']} success"
    
    # Position annotation smartly (avoid overlap)
    if threshold == 50:
        x_pos = batch_num - 100
        y_pos = threshold + 8
        ha = 'right'
    elif threshold == 80:
        x_pos = batch_num + 50
        y_pos = threshold - 8
        ha = 'left'
    elif threshold == 90:
        x_pos = batch_num - 100
        y_pos = threshold - 8
        ha = 'right'
    else:  # 100%
        x_pos = batch_num + 50
        y_pos = threshold - 5
        ha = 'left'
    
    # Create info box
    bbox_info = dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor=color, linewidth=2, alpha=0.95)
    info_text = f'{threshold}% Threshold Crossed\n{window_str}\n{counts_str}'
    text = ax2.text(x_pos, y_pos, info_text,
                   fontsize=9, ha=ha, va='center',
                   bbox=bbox_info, color=color, fontweight='bold',
                   zorder=20)
    
    # Draw arrow to the crossing point
    arrow_props = dict(arrowstyle='->', lw=2, color=color, alpha=0.7)
    ax2.annotate('', xy=(batch_num, threshold),
                xytext=(x_pos, y_pos),
                arrowprops=arrow_props, zorder=19)

ax2.set_xlabel('Batch Number', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax2.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax2.set_ylim(-3, 108)
ax2.set_yticks(np.arange(0, 101, 10))
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
ax2.grid(True, linestyle=':', alpha=0.4, linewidth=1, color='gray')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# =============================================================================
# GRAPH 3: Successful Requests Performance with Degradation Zones
# =============================================================================
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_facecolor('#fafafa')

ax3.text(0.02, 1.08, 'Performance Analysis (Successful Requests Only)', 
         transform=ax3.transAxes, fontsize=20, 
         fontweight='bold', color=COLORS['text_dark'])
ax3.text(0.02, 1.03, 'Response time trends for healthy (200 OK) requests',
        transform=ax3.transAxes, fontsize=11, color='#6b7280', style='italic')

if not healthy.empty:
    # Individual points
    ax3.scatter(healthy['batch_num'], healthy['time_sec'],
               color=COLORS['success'], alpha=0.4, s=15,
               label='Individual Batch Time')
    
    # Rolling average
    healthy_sorted = healthy.sort_values('batch_num')
    healthy_sorted['rolling_avg'] = healthy_sorted['time_sec'].rolling(
        window=ROLLING_WINDOW_SIZE, min_periods=1
    ).mean()
    
    ax3.plot(healthy_sorted['batch_num'], healthy_sorted['rolling_avg'],
            color=COLORS['info'], linewidth=3.5, 
            label=f'{ROLLING_WINDOW_SIZE}-Batch Rolling Avg',
            path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    
    if baseline_avg:
        # Baseline
        ax3.axhline(y=baseline_avg, color=COLORS['baseline'],
                   linestyle='--', linewidth=2.5, label=f'Baseline: {baseline_avg:.3f}s')
        
        # Performance zones with colors
        ax3.axhspan(0, baseline_avg, alpha=0.1, color='green', label='Optimal Zone')
        ax3.axhspan(baseline_avg, baseline_avg * 2, alpha=0.08, color='yellow')
        ax3.axhspan(baseline_avg * 2, baseline_avg * 3, alpha=0.08, color='orange')
        
        # 2x line
        ax3.axhline(y=baseline_avg * 2, color=COLORS['warning'],
                   linestyle=':', linewidth=2, alpha=0.7,
                   label=f'2x Baseline (WARNING)')
        
        # 3x line
        ax3.axhline(y=baseline_avg * 3, color=COLORS['critical'],
                   linestyle=':', linewidth=2, alpha=0.7,
                   label=f'3x Baseline (CRITICAL)')
        
        # Zone labels
        mid_x = ax3.get_xlim()[1] * 0.02
        ax3.text(mid_x, baseline_avg * 0.5, 'OPTIMAL',
                fontsize=10, alpha=0.6, fontweight='bold', color='green')
        ax3.text(mid_x, baseline_avg * 1.5, 'DEGRADED',
                fontsize=10, alpha=0.6, fontweight='bold', color='orange')
        ax3.text(mid_x, baseline_avg * 2.5, 'CRITICAL',
                fontsize=10, alpha=0.6, fontweight='bold', color='red')

ax3.set_xlabel('Batch Number', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax3.set_ylabel('API Response Time (seconds)', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax3.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='gray', fancybox=True, ncol=2)
ax3.grid(True, linestyle=':', alpha=0.4, linewidth=1, color='gray')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# =============================================================================
# GRAPH 4: Status Code Distribution with Enhanced Visuals
# =============================================================================
ax4 = fig.add_subplot(gs[3, 0])
ax4.set_facecolor('#fafafa')

ax4.text(0.02, 1.12, 'HTTP Status Code Distribution', 
         transform=ax4.transAxes, fontsize=20, 
         fontweight='bold', color=COLORS['text_dark'])
ax4.text(0.02, 1.05, 'Breakdown of response types received during test',
        transform=ax4.transAxes, fontsize=11, color='#6b7280', style='italic')

status_counts = df['status_code'].value_counts().sort_index()
total_requests = len(df)

# Create bars with gradient effect
colors = [COLORS['success'] if code == 200 else COLORS['failure'] 
          for code in status_counts.index]

bars = ax4.bar([str(code) for code in status_counts.index],
              status_counts.values, color=colors, alpha=0.8,
              edgecolor='black', linewidth=2, width=0.6)

# Add 3D-like shadow effect
for bar in bars:
    shadow = FancyBboxPatch((bar.get_x() + 0.02, 0),
                           bar.get_width(), bar.get_height(),
                           boxstyle="round,pad=0.01", 
                           edgecolor='none', facecolor='gray',
                           alpha=0.2, zorder=0)
    ax4.add_patch(shadow)

# Enhanced labels with icons
for i, (bar, code, count) in enumerate(zip(bars, status_counts.index, status_counts.values)):
    percentage = (count / total_requests) * 100
    height = bar.get_height()
    
    # Status symbol
    symbol = '✓' if code == 200 else '✗'
    
    # Label above bar
    label_text = f'{symbol} {code}\n{count:,} requests\n({percentage:.1f}%)'
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=bar.get_facecolor(), linewidth=2.5, alpha=0.98)
    ax4.text(bar.get_x() + bar.get_width()/2, height + (ax4.get_ylim()[1] * 0.03),
            label_text, ha='center', va='bottom', fontsize=11,
            fontweight='bold', bbox=bbox_props,
            color=bar.get_facecolor())

ax4.set_xlabel('HTTP Status Code', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax4.set_ylabel('Number of Requests', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax4.grid(True, axis='y', linestyle=':', alpha=0.4, linewidth=1, color='gray')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# =============================================================================
# GRAPH 5: Timeline View with Time-Based Insights
# =============================================================================
ax5 = fig.add_subplot(gs[4, 0])
ax5.set_facecolor('#fafafa')

ax5.text(0.02, 1.08, 'Minute-by-Minute Success/Failure Timeline', 
         transform=ax5.transAxes, fontsize=20, 
         fontweight='bold', color=COLORS['text_dark'])
ax5.text(0.02, 1.03, 'Temporal distribution of batch outcomes',
        transform=ax5.transAxes, fontsize=11, color='#6b7280', style='italic')

x_positions = np.arange(len(minute_summary))
bar_width = 0.75

# Create stacked bars with edge emphasis
bars_success = ax5.bar(x_positions, minute_summary['successes'],
                      bar_width, label='Success (200 OK)',
                      color=COLORS['success'], alpha=0.85,
                      edgecolor='darkgreen', linewidth=1.5)

bars_failure = ax5.bar(x_positions, minute_summary['failures'],
                      bar_width, bottom=minute_summary['successes'],
                      label='Failure (Non-200)',
                      color=COLORS['failure'], alpha=0.85,
                      edgecolor='darkred', linewidth=1.5)

# Add total count on top of bars
for i, (success, failure) in enumerate(zip(minute_summary['successes'], 
                                           minute_summary['failures'])):
    total = success + failure
    if total > 0:
        # Show failure count if there are failures
        if failure > 0:
            ax5.text(i, success + failure + (ax5.get_ylim()[1] * 0.02),
                    f'{int(failure)}',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=COLORS['failure'])

time_labels = [t.strftime('%H:%M') for t in minute_summary.index]
ax5.set_xticks(x_positions)
ax5.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=10)

ax5.set_xlabel('Time (HH:MM)', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax5.set_ylabel('Number of Batches', fontsize=14, fontweight='bold', color=COLORS['text_dark'])
ax5.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
ax5.grid(True, axis='y', linestyle=':', alpha=0.4, linewidth=1, color='gray')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# =============================================================================
# GRAPH 6: Executive Summary with Professional Layout
# =============================================================================
ax6 = fig.add_subplot(gs[5, 0])
ax6.axis('off')

# Create a professional summary box
summary_box = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                            boxstyle="round,pad=0.02",
                            transform=ax6.transAxes,
                            edgecolor=COLORS['info'],
                            facecolor='white',
                            linewidth=3,
                            zorder=0)
ax6.add_patch(summary_box)

# Header
ax6.text(0.5, 0.88, '📊 EXECUTIVE SUMMARY', transform=ax6.transAxes,
        fontsize=18, fontweight='bold', ha='center',
        color=COLORS['text_dark'])

# Metrics in columns
col1_x, col2_x, col3_x = 0.08, 0.37, 0.68
current_y = 0.75

# Calculate metrics
total_api_time = sum(df['time_sec'])
total_time_min = total_api_time / 60
success_count = len(healthy)
failure_count = len(unhealthy)
success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0

# Column 1: Volume
ax6.text(col1_x, current_y, '📦 VOLUME PROCESSED', transform=ax6.transAxes,
        fontsize=12, fontweight='bold', color=COLORS['info'])
metrics_text = f"""
Batches: {summary.get('total_batches_run', 0):,}
Transactions: {summary.get('total_transactions_processed', 0):,}
Duration: {total_time_min:.1f} min
"""
ax6.text(col1_x, current_y - 0.05, metrics_text, transform=ax6.transAxes,
        fontsize=10, va='top', family='monospace', color=COLORS['text_dark'])

# Column 2: Performance
ax6.text(col2_x, current_y, '⚡ PERFORMANCE', transform=ax6.transAxes,
        fontsize=12, fontweight='bold', color=COLORS['info'])
perf_text = f"""
Baseline: {f'{baseline_avg:.3f}s' if baseline_avg else 'N/A'}
Success Rate: {success_rate:.1f}%
Successful: {success_count:,}
Failed: {failure_count:,}
"""
ax6.text(col2_x, current_y - 0.05, perf_text, transform=ax6.transAxes,
        fontsize=10, va='top', family='monospace', color=COLORS['text_dark'])

# Column 3: Failure Analysis
ax6.text(col3_x, current_y, '🔍 FAILURE POINT', transform=ax6.transAxes,
        fontsize=12, fontweight='bold', color=COLORS['info'])
failure_text = f"""
First Error: Batch {first_error_batch if first_error_batch else 'N/A'}
Txns Before Fail: {txns_until_failure:,}
Error Type: {milestones.get('first_error_message', 'N/A')[:20]}...
"""
ax6.text(col3_x, current_y - 0.05, failure_text, transform=ax6.transAxes,
        fontsize=10, va='top', family='monospace', color=COLORS['text_dark'])

# Key Finding Box
finding_y = 0.32
finding_box = FancyBboxPatch((0.06, finding_y - 0.02), 0.88, 0.25,
                            boxstyle="round,pad=0.01",
                            transform=ax6.transAxes,
                            edgecolor=COLORS['warning'],
                            facecolor='#fffbeb',
                            linewidth=2.5,
                            zorder=1)
ax6.add_patch(finding_box)

ax6.text(0.5, finding_y + 0.20, '💡 KEY FINDINGS', transform=ax6.transAxes,
        fontsize=12, fontweight='bold', ha='center', color=COLORS['warning'])

# Generate intelligent finding
if first_error_batch:
    finding_text = f"""
The API processed {txns_until_failure:,} transactions successfully before the first failure
at batch {first_error_batch:,}. The rolling error rate then progressively increased through
{', '.join([f"{t['threshold']}%" for t in thresholds_data])} thresholds, indicating memory
exhaustion rather than random failures. This pattern suggests the pod reached its
capacity limit after processing approximately {txns_until_failure//1000}K transactions.
"""
else:
    finding_text = "No failures detected during the test period."

ax6.text(0.5, finding_y + 0.09, finding_text.strip(), transform=ax6.transAxes,
        fontsize=10, ha='center', va='top', wrap=True,
        color=COLORS['text_dark'], style='italic')

# Footer
ax6.text(0.5, 0.02, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        transform=ax6.transAxes, fontsize=9, ha='center',
        color='#9ca3af', style='italic')

# Save high-res output
output_path = 'stress_test_output/stakeholder_dashboard_professional.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"\n✅ Professional dashboard saved to: {output_path}")
print(f"   Resolution: 300 DPI")
print(f"   Style: Modern Corporate")
print(f"   Annotations: {len(thresholds_data)} intelligent threshold markers added")

plt.tight_layout(rect=[0, 0.01, 1, 0.99])
plt.show()

print("\n" + "="*80)
print("✅ DASHBOARD GENERATION COMPLETE")
print("="*80)
