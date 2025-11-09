import pandas as pd
import requests
import json
import time
import signal
from typing import Dict, List, Any
import os
import sys

# --- CONFIGURATION ---
API_URL = 'https/your-domain.cloud.databricks.com/serving-endpoints/us__spacy_ner/invocations'
API_NAME = 'us_spacy_ner_stress_test'
BATCH_SIZE = 50
OUTPUT_FOLDER = 'stress_test_output'

# --- INVESTIGATION & STOP CONDITIONS ---
ROLLING_WINDOW_SIZE = 20  # Check the last 20 batches
STOP_FAILURE_RATE = 1.0   # Stop if rolling rate hits 100%
STOP_CONSECUTIVE_FAILURES = 10 # Backup: stop after 10 failures in a row

# --- LOGGING & PRINTING ---
PRINT_INTERVAL_SECONDS = 300 # Print status every 5 minutes
INVESTIGATION_THRESHOLDS = [0.5, 0.8, 0.9, 1.0] # Log samples at 50%, 80%, 90%, 100%
INVESTIGATION_SAMPLES = 3 # Save 3 examples of 200s and 500s per threshold
# -------------------------

class StressTestInvestigator:
    """
    This script is an "Investigator," not a "Timer."
    It's designed to capture evidence of "flapping" (200/500 interchange)
    and intelligently stop when the service is confirmed to be unstable.
    """
    
    def __init__(self, api_name: str, api_url: str, batch_size: int, output_folder: str):
        self.api_name = api_name
        self.api_url = api_url
        self.batch_size = batch_size
        self.output_folder = output_folder
        
        # Use a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # This is the NEW "Evidence Locker" - it replaces all the old lists
        self.results_per_batch: List[Dict[str, Any]] = []
        
        # This is the NEW "Smoking Gun" log
        self.investigation_log_path = os.path.join(self.output_folder, 'investigation_log.csv')
        self.investigation_log_headers_written = False
        self.investigation_thresholds_hit = set() # Tracks 50%, 80% etc.

        # State tracking
        self.baseline_avg = None
        self.last_degradation_severity = None
        self.running = True
        self.consecutive_failures = 0

    # ---
    # We are KEEPING your two functions exactly as they were.
    # They are perfect for this test.
    # ---
    def generate_unique_transaction(self, iteration: int, num_unique_tokens: int = 50) -> str:
        """
        Generate truly unique transaction with no duplicates.
        Each field uses iteration number to guarantee uniqueness.
        (This is your exact code, it's perfect)
        """
        transaction_types = [
            "POS", "ATM", "ONLINE", "TRANSFER", "PAYMENT",
            "REFUND", "WITHDRAWAL", "DEPOSIT"
        ]
        merchants = [
            "AMAZON", "WALMART", "STARBUCKS", "SHELL", "MCDONALDS",
            "TARGET", "COSTCO", "BESTBUY", "NETFLIX", "UBER",
            "AIRBNB", "BOOKING", "PAYPAL", "VENMO", "SQUARE",
            "SPOTIFY", "APPLE", "GOOGLE", "MICROSOFT", "CHIPOTLE"
        ]
        
        txn_id = f"TXN{iteration:010d}"
        merchant_base = merchants[iteration % len(merchants)]
        merchant = f"{merchant_base}{iteration}"
        
        amount = f"${(iteration % 995) + 5.0 + (iteration * 0.01) % 1:.2f}"
        card = f"CARD-{1000 + iteration}"
        acct = f"ACCT{1000000 + iteration}"
        auth = f"AUTH{100000 + iteration}"
        ref = f"REF{iteration}{iteration % 10000}"
        merchant_id = f"MID{100000 + iteration}"
        terminal = f"T{1000 + iteration}"
        batch = f"B{100 + (iteration % 900)}"
        trans_type = transaction_types[iteration % len(transaction_types)]
        
        unique_tokens = []
        remaining = num_unique_tokens - 11
        
        for i in range(remaining):
            token_type = ['LOC', 'ID', 'CODE', 'SEQ'][i % 4]
            token = f"{token_type}{iteration}{i}"
            unique_tokens.append(token)
            
        description_parts = [
            txn_id, trans_type, merchant, amount, card, acct, auth,
            ref, merchant_id, terminal, batch
        ] + unique_tokens
        
        return " ".join(description_parts)

    def build_tensor_payload(self, descriptions: List[str], memos: List[str]) -> Dict:
        """
        Build tensor-based payload for batch inference.
        (This is your exact code)
        """
        batch_size = len(descriptions)
        payload = {
            "inputs": [
                {
                    "name": "description",
                    "datatype": "BYTES",
                    "shape": [batch_size, 1],
                    "data": descriptions
                },
                {
                    "name": "memo",
                    "datatype": "BYTES",
                    "shape": [batch_size, 1],
                    "data": memos
                }
            ]
        }
        return payload

    # ---
    # This is the FIRST MAJOR CHANGE: The "Investigator" send function
    # ---
    def send_batch_request(self, descriptions: List[str], memos: List[str]) -> dict:
        """
        Sends the batch and captures all investigative evidence.
        Returns a dictionary (our "evidence report" for this batch).
        """
        payload = self.build_tensor_payload(descriptions, memos)
        api_time = 0.0
        status_code = 0
        response_body = None  # This will hold the 200 JSON or the 500 Error
        
        try:
            api_start = time.time()
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=120  # 2-minute timeout
            )
            api_time = time.time() - api_start
            status_code = response.status_code

            # This is the "Smoking Gun" capture!
            try:
                # Try to parse the JSON (works for 200 OK and 500 Error)
                response_body = response.json()
            except requests.exceptions.JSONDecodeError:
                # If it's not JSON (e.g., a 502 Bad Gateway HTML), just get raw text
                response_body = {"error": response.text[:200]} # Get first 200 chars

        except Exception as e:
            api_time = time.time() - (api_start if 'api_start' in locals() else time.time())
            status_code = 0 # 0 = Connection Error / Timeout
            response_body = {"error": str(e)}

        # Return the full evidence report
        return {
            "time_sec": api_time,
            "status_code": status_code,
            "response_body": response_body
        }

    # ---
    # SECOND MAJOR CHANGE: This function now reads from our "Evidence Locker"
    # ---
    def detect_degradation(self, batch_num: int, batch_time: float):
        """
        Detect performance degradation by comparing to baseline.
        Print only when severity changes.
        """
        # Step 1: Establish baseline
        if self.baseline_avg is None:
            if batch_num >= 10:
                # Get all successful times from the first 10 batches
                successful_times = [
                    r['time_sec'] for r in self.results_per_batch
                    if r['status_code'] == 200 and r['batch_num'] <= 10
                ]
                if successful_times:
                    self.baseline_avg = sum(successful_times) / len(successful_times)
                    print(f"\n✓ Baseline established: {self.baseline_avg:.3f}s avg")
                else:
                    print("\n✗ Could not establish baseline, no successful requests in first 10 batches.")
            return

        # Step 2: Check for degradation
        degradation_factor = batch_time / self.baseline_avg
        
        if degradation_factor >= 3.0:
            severity = 'CRITICAL'
            if self.last_degradation_severity != severity:
                print(f"\n⚠️ CRITICAL Degradation: Batch {batch_num} took {batch_time:.3f}s ({degradation_factor:.1f}x baseline)")
                self.last_degradation_severity = severity
        
        elif degradation_factor >= 2.0:
            severity = 'WARNING'
            if self.last_degradation_severity not in ['WARNING', 'CRITICAL']:
                print(f"\n⚠️ WARNING Degradation: Batch {batch_num} took {batch_time:.3f}s ({degradation_factor:.1f}x baseline)")
                self.last_degradation_severity = severity
        else:
            if self.last_degradation_severity is not None:
                print(f"\n✓ Performance recovered at batch {batch_num}: {batch_time:.3f}s")
                self.last_degradation_severity = None
                
    # ---
    # NEW FUNCTION: This is your "Investigation Mode" sampler
    # ---
    def log_investigation_samples(self, triggering_batch: int, window_start: int, error_rate: float):
        """
        Saves 200 OK and 500 Error samples to a CSV log
        to prove what's happening during an "interchange".
        """
        # Find up to 3 successful and 3 failed samples from the window
        recent_results = self.results_per_batch[window_start-1:triggering_batch]
        
        success_samples = [r for r in recent_results if r['status_code'] == 200]
        failure_samples = [r for r in recent_results if r['status_code'] != 200]

        # We'll save 3 of each, or fewer if not available
        num_samples = min(len(success_samples), len(failure_samples), INVESTIGATION_SAMPLES)
        if num_samples == 0:
             # If it's all one or the other, just log one sample
             num_samples = INVESTIGATION_SAMPLES 

        log_entries = []
        for i in range(num_samples):
            # Get success sample
            if i < len(success_samples):
                sample = success_samples[i]
                log_entries.append({
                    'triggering_batch_num': triggering_batch,
                    'window_range': f"{window_start}-{triggering_batch}",
                    'error_rate_percent': int(error_rate * 100),
                    'sample_type': '200_OK',
                    'sample_batch_num': sample['batch_num'],
                    'sample_output': json.dumps(sample['response_body'])
                })
            
            # Get failure sample
            if i < len(failure_samples):
                sample = failure_samples[i]
                log_entries.append({
                    'triggering_batch_num': triggering_batch,
                    'window_range': f"{window_start}-{triggering_batch}",
                    'error_rate_percent': int(error_rate * 100),
                    'sample_type': 'NON-200_FAILURE',
                    'sample_batch_num': sample['batch_num'],
                    'sample_output': json.dumps(sample['response_body'])
                })

        if not log_entries:
            return

        # Write these new entries to the CSV file
        df = pd.DataFrame(log_entries)
        if not self.investigation_log_headers_written:
            df.to_csv(self.investigation_log_path, index=False, mode='w', header=True)
            self.investigation_log_headers_written = True
        else:
            df.to_csv(self.investigation_log_path, index=False, mode='a', header=False)

    # ---
    # FINAL MAJOR CHANGE: The main loop with "Intelligent Stop"
    # ---
    def run_stress_test(self):
        print(f"\n{'='*80}")
        print(f"STARTING STRESS TEST: {self.api_name}")
        print(f"{'='*80}")
        print(f"URL: {self.api_url}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Output Folder: {self.output_folder}")
        print(f"Will stop if rolling {ROLLING_WINDOW_SIZE}-batch error rate hits 100%")
        print("Press Ctrl+C to stop\n")

        os.makedirs(self.output_folder, exist_ok=True)
        
        batch_num = 0
        last_print_time = time.time()
        stop_reason = "Test manually stopped (Ctrl+C)"

        def signal_handler(sig, frame):
            print("\n\nReceived stop signal. Halting...")
            self.running = False
            nonlocal stop_reason
            stop_reason = "Test manually stopped (Ctrl+C)"

        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.running:
                batch_num += 1
                
                descriptions = []
                memos = []
                for i in range(self.batch_size):
                    iteration = (batch_num - 1) * self.batch_size + i + 1
                    description = self.generate_unique_transaction(iteration)
                    descriptions.append(description)
                    memos.append("")

                # 1. Get the full evidence report
                result_dict = self.send_batch_request(descriptions, memos)
                result_dict['batch_num'] = batch_num
                
                # 2. Add it to our "Evidence Locker"
                self.results_per_batch.append(result_dict)
                
                # 3. Handle success or failure
                if result_dict['status_code'] == 200:
                    self.consecutive_failures = 0
                    self.detect_degradation(batch_num, result_dict['time_sec'])
                else:
                    self.consecutive_failures += 1
                    # Minimal printing: only print the error once
                    if self.consecutive_failures == 1:
                         print(f"\n❌ FIRST FAILURE at batch {batch_num}: Status {result_dict['status_code']}, Error: {result_dict['error_msg']}")

                # 4. Intelligent Stop & Investigation Logic
                if batch_num >= ROLLING_WINDOW_SIZE:
                    window_start = (batch_num - ROLLING_WINDOW_SIZE) + 1
                    recent_results = self.results_per_batch[-ROLLING_WINDOW_SIZE:]
                    failure_count = sum(1 for r in recent_results if r['status_code'] != 200)
                    error_rate = failure_count / ROLLING_WINDOW_SIZE
                    
                    # Check if we crossed a new threshold
                    for threshold in INVESTIGATION_THRESHOLDS:
                        if error_rate >= threshold and threshold not in self.investigation_thresholds_hit:
                            print(f"\nINVESTIGATION: Rolling error rate hit {int(threshold*100)}% (Window: {window_start}-{batch_num})")
                            self.log_investigation_samples(batch_num, window_start, error_rate)
                            self.investigation_thresholds_hit.add(threshold)
                            
                            if threshold == STOP_FAILURE_RATE:
                                print(f"💥 HALTING: Rolling error rate is 100%. Service is unresponsive.")
                                stop_reason = f"Rolling error rate hit 100%"
                                self.running = False

                # Backup stop condition
                if self.consecutive_failures >= STOP_CONSECUTIVE_FAILURES:
                    print(f"\n💥 HALTING: {STOP_CONSECUTIVE_FAILURES} consecutive failures.")
                    stop_reason = f"{STOP_CONSECUTIVE_FAILURES} consecutive failures"
                    self.running = False
                
                # 5. Minimal 5-min status print
                current_time = time.time()
                if current_time - last_print_time >= PRINT_INTERVAL_SECONDS:
                    total_time_api = sum(r['time_sec'] for r in self.results_per_batch)
                    avg_time = total_time_api / batch_num if batch_num > 0 else 0
                    print(f"--- Status [Batch {batch_num:6d} | Txns: {batch_num * self.batch_size:10,}] | API Avg: {avg_time:.3f}s ---")
                    last_print_time = current_time

        except Exception as e:
            print(f"\n\n❌ UNEXPECTED FATAL ERROR: {str(e)}")
            stop_reason = f"Fatal script error: {str(e)}"

        finally:
            self.save_final_metrics(stop_reason)
            self.session.close()

    def save_final_metrics(self, stop_reason: str):
        """Saves the complete "Evidence Locker" to a JSON file."""
        print(f"\n{'='*80}")
        print("STRESS TEST COMPLETED")
        print(f"Stop Reason: {stop_reason}")
        print("Saving final metrics...")
        
        total_batches = len(self.results_per_batch)
        if total_batches == 0:
            print("No data captured. Exiting.")
            return

        total_transactions = total_batches * self.batch_size
        
        # We only want the core data for the JSON, not the full response bodies
        # The response bodies are in the `investigation_log.csv`
        metrics_for_json = []
        for r in self.results_per_batch:
            metrics_for_json.append({
                "batch_num": r['batch_num'],
                "time_sec": r['time_sec'],
                "status_code": r['status_code'],
                "error_msg": r['response_body'].get('error') if isinstance(r['response_body'], dict) else str(r['response_body'])
            })
        
        first_error_result = next((r for r in metrics_for_json if r['status_code'] != 200), None)

        final_data = {
            'test_summary': {
                'api_name': self.api_name,
                'api_url': self.api_url,
                'total_batches_run': total_batches,
                'total_transactions_processed': total_transactions,
                'baseline_avg_time': self.baseline_avg,
                'stop_reason': stop_reason
            },
            'failure_milestones': {
                'first_error_batch_num': first_error_result['batch_num'] if first_error_result else None,
                'total_transactions_at_first_error': (first_error_result['batch_num'] * self.batch_size) if first_error_result else None,
                'first_error_message': first_error_result['error_msg'] if first_error_result else None
            },
            'results_per_batch': metrics_for_json # This is the full "Evidence Locker"
        }

        metrics_path = os.path.join(self.output_folder, 'stress_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_data, f, indent=2)

        print(f"\n✅ Final metrics saved to: {metrics_path}")
        if self.investigation_log_headers_written:
            print(f"✅ Investigation log saved to: {self.investigation_log_path}")

def main():
    print("="*80)
    print("API STRESS TEST (INVESTIGATOR SCRIPT v2)")
    print("="*80)
    print(f"\nAPI URL: {API_URL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print(f"\nThis test will run until:")
    print(f"  1. You press Ctrl+C")
    print(f"  2. The rolling {ROLLING_WINDOW_SIZE}-batch error rate hits 100%")
    print(f"  3. The API returns {STOP_CONSECUTIVE_FAILURES} consecutive failures")
    
    try:
        input("\nPress Enter to start...")
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        sys.exit(0)

    stress_tester = StressTestInvestigator(API_NAME, API_URL, BATCH_SIZE, OUTPUT_FOLDER)
    stress_tester.run_stress_test()

    print(f"\nRun 'python graph_stress_test.py {OUTPUT_FOLDER}/stress_metrics.json' to generate graphs.")

if __name__ == "__main__":
    main()
