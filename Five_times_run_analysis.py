import json
import statistics
from pathlib import Path
import argparse
from scipy import stats  # For the t-test

def load_run_data(json_path: Path) -> dict:
    """Loads and processes data from a single results.json file."""
    if not json_path.exists():
        print(f"Error: File not found at {json_path}")
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    batch_times = data.get('batch_times')
    if not batch_times:
        print(f"Error: 'batch_times' key missing or empty in {json_path}")
        return None
        
    num_batches = len(batch_times)
    total_time = sum(batch_times)
    avg_time = statistics.mean(batch_times) 
    
    return {
        "path": str(json_path),
        "num_batches": num_batches,
        "total_time_sec": total_time,
        "total_time_min": total_time / 60.0,
        "avg_batch_time_sec": avg_time,
        "median_batch_time_sec": statistics.median(batch_times),
        "stdev_batch_time": statistics.stdev(batch_times),
        "all_batch_times": batch_times,
    }

def print_section_header(title):
    """Prints a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title.upper()} ".center(60))
    print("=" * 60)

def analyze_performance(output_folder: str):
    """
    Analyzes performance data from the 'without_zone' and 'with_zone'
    experiment subfolders.
    """
    base_path = Path(output_folder)
    if not base_path.is_dir():
        print(f"Error: '{output_folder}' is not a valid directory.")
        return

    all_without_runs = []
    all_with_runs = []

    print(f"Loading data from '{output_folder}'...")
    
    # --- 1. Load all data from the 5 folders ---
    for i in range(1, 6):
        without_path = base_path / f"without_zone_{i}" / "results.json"
        without_data = load_run_data(without_path)
        if without_data:
            all_without_runs.append(without_data)
            
        with_path = base_path / f"with_zone_{i}" / "results.json"
        with_data = load_run_data(with_path)
        if with_data:
            all_with_runs.append(with_data)
            
    if not all_without_runs or not all_with_runs:
        print("Error: Could not load data for both 'with_zone' and 'without_zone'. Aborting.")
        return

    # --- 2. Per-Experiment Analysis ---
    print_section_header("Per-Experiment Analysis")
    
    for i in range(len(all_without_runs)):
        without_run = all_without_runs[i]
        with_run = all_with_runs[i]
        
        print(f"\n--- Run {i+1} (Batches: {with_run['num_batches']}) ---")
        
        cost_per_run_sec = with_run['total_time_sec'] - without_run['total_time_sec']
        cost_per_run_min = cost_per_run_sec / 60.0
        avg_cost_per_batch = with_run['avg_batch_time_sec'] - without_run['avg_batch_time_sec']

        print(f"  without_zone | Avg: {without_run['avg_batch_time_sec']:.6f}s | Total: {without_run['total_time_min']:.2f} mins")
        print(f"  with_zone    | Avg: {with_run['avg_batch_time_sec']:.6f}s | Total: {with_run['total_time_min']:.2f} mins")
        print(f"  COST (Run {i+1}) | {cost_per_run_sec:.2f} seconds ({cost_per_run_min:.2f} minutes) slower.")

        cost_from_avg = avg_cost_per_batch * with_run['num_batches']
        print(f"  Verification | Cost from Totals: {cost_per_run_sec:.6f}s")
        print(f"               | Cost from Avgs:   {cost_from_avg:.6f}s")
        print(f"               | Difference:     {(cost_per_run_sec - cost_from_avg):.10f}s (should be 0)")


    # --- 3. Overall Analysis (Aggregating all 5 runs) ---
    print_section_header("Overall Analysis (All 5 Experiments)")

    all_without_batches = [t for run in all_without_runs for t in run['all_batch_times']]
    all_with_batches = [t for run in all_with_runs for t in run['all_batch_times']]
    
    if not all_without_batches or not all_with_batches:
        print("Error: Aggregated batch lists are empty. Cannot continue.")
        return
        
    total_batches_all_runs = len(all_with_batches)
    
    overall_total_without_time_sec = sum(all_without_batches)
    overall_total_with_time_sec = sum(all_with_batches)
    
    overall_avg_without_sec = statistics.mean(all_without_batches)
    overall_avg_with_sec = statistics.mean(all_with_batches)
    
    overall_median_without_sec = statistics.median(all_without_batches)
    overall_median_with_sec = statistics.median(all_with_batches)

    # --- 4. Final Cost and Performance Calculations ---
    print(f"Total Batches Processed (5 runs x {total_batches_all_runs / 5}): {total_batches_all_runs}")

    print("\n--- Overall Performance Stats ---")
    print(f"  without_zone | Avg: {overall_avg_without_sec:.6f}s | Median: {overall_median_without_sec:.6f}s")
    print(f"  with_zone    | Avg: {overall_avg_with_sec:.6f}s | Median: {overall_median_with_sec:.6f}s")
    
    print("\n--- Overall Cost & Overhead ---")
    
    total_cost_overhead_sec = overall_total_with_time_sec - overall_total_without_time_sec
    total_cost_overhead_min = total_cost_overhead_sec / 60.0
    
    avg_cost_per_batch_sec = overall_avg_with_sec - overall_avg_without_sec
    
    print(f"Total Cost Overhead (5 Runs): {total_cost_overhead_sec:.2f} seconds")
    print(f"                            = {total_cost_overhead_min:.2f} minutes")
    print(f"Average Cost Per Batch:       {avg_cost_per_batch_sec:.6f} seconds")

    total_cost_from_avg = avg_cost_per_batch_sec * total_batches_all_runs
    print("\n--- Final Verification (Overall) ---")
    print(f"  A. Cost from Total Times: {total_cost_overhead_sec:.6f}s")
    print(f"  B. Cost from Avg * Batches: {total_cost_from_avg:.6f}s")
    print(f"  Difference (A - B):     {(total_cost_overhead_sec - total_cost_from_avg):.10f}s (should be 0)")
    
    times_faster = overall_avg_with_sec / overall_avg_without_sec
    percent_faster = (times_faster - 1) * 100.0
    
    print("\n--- Final Conclusion ---")
    print(f"'without_zone' is {times_faster:.3f}x times faster (or {percent_faster:.1f}% faster) than 'with_zone' on average.")


    # --- 5. Statistical Significance Test (t-test) ---
    # This section is modified with non-technical explanations
    # ---------------------------------------------------
    print_section_header("Statistical Significance (t-test)")
    print("This test proves if the slowdown is real or just a random fluke.")

    t_statistic, p_value = stats.ttest_ind(
        all_with_batches, 
        all_without_batches, 
        equal_var=False
    )

    print(f"\n  p-value: {p_value:0.5e}")
    print("     (This number is the 'random chance' probability.)")
    print("     (A tiny number means the result is NOT random.)")


    print("\n--- t-test Verdict (in plain English) ---")
    
    # We check if p_value is less than 0.001 (a common standard for "extremely significant")
    if p_value < 0.001:
        print("✓ VERDICT: The difference is REAL and NOT RANDOM.")
        print(f"\n  The probability of this slowdown being a random fluke")
        print(f"  is {p_value:0.1e} (practically zero).")
        print("\n  This is definitive statistical proof that the 'with_zone' API")
        print("  is indeed consistently slower. The lag is a real, predictable cost.")
        
    # We check if p_value is less than 0.05 (the standard for "significant")
    elif p_value < 0.05:
        print("✓ VERDICT: The difference is VERY LIKELY REAL.")
        print(f"\n  The 'random chance' probability is less than 5% (p = {p_value:.4f}).")
        print("  The evidence strongly suggests the slowdown is a real effect")
        print("  and not just a random fluke.")
        
    # If p_value is 0.05 or higher
    else:
        print("✗ VERDICT: The difference could be RANDOM CHANCE.")
        print(f"\n  The 'random chance' probability is high (p = {p_value:.4f}).")
        print("  We cannot prove that the 'with_zone' API is consistently slower.")
        print("  The difference we measured might just be random noise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze 'with_zone' vs 'without_zone' performance."
    )
    parser.add_argument(
        "output_folder", 
        type=str, 
        help="The path to the main 'output' folder containing the 10 experiment subfolders (e.g., 'my_output_dir')."
    )
    
    args = parser.parse_args()
    analyze_performance(args.output_folder)

