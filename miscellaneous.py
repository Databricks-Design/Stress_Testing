def main():
    
    INPUT_CSV = 'input_data.csv'
    API_URL = 'PLACEHOLDER_API_URL'  # Single API endpoint
    BATCH_SIZE = 50
    ROWS_PER_FILE = 100000
    NUM_RUNS = 5  # Run 5 times
    
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV, usecols=['description', 'memo'])
    print(f"Loaded {len(df)} transactions")
    
    all_run_summaries = []
    
    # Run the test 5 times
    for run_num in range(1, NUM_RUNS + 1):
        output_folder = f'output/without_zone_{run_num}'
        
        print("\n" + "="*60)
        print(f"RUN {run_num}/{NUM_RUNS}: WITHOUT_ZONE_{run_num}")
        print("="*60)
        
        tester = APITester(f'without_zone_{run_num}', API_URL, BATCH_SIZE)
        summary = tester.run_test(df, output_folder, ROWS_PER_FILE)
        
        all_run_summaries.append(summary)
        
        # Save metrics
        results_json_path = os.path.join(output_folder, 'results.json')
        with open(results_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved: {results_json_path}")
        
        # Highlight average API time for this run
        print(f"\n*** AVERAGE API TIME FOR RUN {run_num}: {summary['avg_batch_time']:.2f}s per batch ***\n")
        
        # Print sample outputs
        print_sample_outputs(output_folder)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    
    # Print summary of all runs
    print("\nSUMMARY ACROSS ALL RUNS:")
    print("-" * 60)
    for idx, summary in enumerate(all_run_summaries, 1):
        print(f"Run {idx}: Avg API time = {summary['avg_batch_time']:.2f}s per batch")
    
    overall_avg = sum(s['avg_batch_time'] for s in all_run_summaries) / len(all_run_summaries)
    print("-" * 60)
    print(f"OVERALL AVERAGE API TIME: {overall_avg:.2f}s per batch")
    print("=" * 80)


if __name__ == "__main__":
    main()
