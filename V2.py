import pandas as pd
import requests
import json
import time
import signal
from typing import Dict, List
import os

class StressTestRunner:
    """
    Stress test API until failure, tracking performance degradation.
    Generates unique sequential transactions for maximum vocab growth.
    """
    
    def __init__(self, api_name: str, api_url: str, batch_size: int = 50):
        self.api_name = api_name
        self.api_url = api_url
        self.batch_size = batch_size
        self.headers = {'Content-Type': 'application/json'}
        
        self.batch_times = []
        self.batch_numbers = []
        self.status_codes = []
        self.errors = []
        self.degradation_events = []
        
        self.baseline_avg = None
        self.last_degradation_severity = None
        self.running = True
        
    def generate_unique_transaction(self, iteration: int, num_unique_tokens: int = 50) -> str:
        """
        Generate truly unique transaction with no duplicates.
        Each field uses iteration number to guarantee uniqueness.
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
            txn_id,
            trans_type,
            merchant,
            amount,
            card,
            acct,
            auth,
            ref,
            merchant_id,
            terminal,
            batch
        ] + unique_tokens
        
        return " ".join(description_parts)
    
    def build_tensor_payload(self, descriptions: List[str], memos: List[str]) -> Dict:
        """Build tensor-based payload for batch inference."""
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
    
    def send_batch_request(self, descriptions: List[str], memos: List[str]) -> tuple:
        """Send batch request and measure only API time."""
        payload = self.build_tensor_payload(descriptions, memos)
        
        try:
            api_start = time.time()
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            
            api_time = time.time() - api_start
            
            return api_time, response.status_code, None
                
        except Exception as e:
            return 0.0, 0, str(e)
    
    def detect_degradation(self, batch_num: int, batch_time: float):
        """
        Detect performance degradation by comparing to baseline.
        Print only when severity changes.
        """
        if batch_num == 10:
            self.baseline_avg = sum(self.batch_times[:10]) / 10
            print(f"\n✓ Baseline established: {self.baseline_avg:.2f}s avg")
            return
        
        if batch_num < 10:
            return
        
        rolling_window = 10
        if len(self.batch_times) >= rolling_window:
            recent_avg = sum(self.batch_times[-rolling_window:]) / rolling_window
            
            degradation_factor = batch_time / self.baseline_avg
            
            if degradation_factor >= 3.0:
                event = {
                    'batch_num': batch_num,
                    'batch_time': batch_time,
                    'baseline_avg': self.baseline_avg,
                    'recent_avg': recent_avg,
                    'degradation_factor': degradation_factor,
                    'severity': 'CRITICAL',
                    'total_transactions': batch_num * self.batch_size
                }
                self.degradation_events.append(event)
                
                if self.last_degradation_severity != 'CRITICAL':
                    print(f"\n⚠️  CRITICAL DEGRADATION started at batch {batch_num}: {batch_time:.2f}s ({degradation_factor:.1f}x baseline)")
                    self.last_degradation_severity = 'CRITICAL'
                    
            elif degradation_factor >= 2.0:
                event = {
                    'batch_num': batch_num,
                    'batch_time': batch_time,
                    'baseline_avg': self.baseline_avg,
                    'recent_avg': recent_avg,
                    'degradation_factor': degradation_factor,
                    'severity': 'WARNING',
                    'total_transactions': batch_num * self.batch_size
                }
                self.degradation_events.append(event)
                
                if self.last_degradation_severity != 'WARNING' and self.last_degradation_severity != 'CRITICAL':
                    print(f"\n⚠️  WARNING: Degradation started at batch {batch_num}: {batch_time:.2f}s ({degradation_factor:.1f}x baseline)")
                    self.last_degradation_severity = 'WARNING'
            else:
                if self.last_degradation_severity is not None:
                    print(f"\n✓ Performance recovered at batch {batch_num}: {batch_time:.2f}s")
                    self.last_degradation_severity = None
    
    def save_checkpoint(self, batch_num: int, output_folder: str):
        """Save checkpoint metrics."""
        checkpoint_data = {
            'batch_numbers': self.batch_numbers,
            'batch_times': self.batch_times,
            'status_codes': self.status_codes,
            'errors': self.errors,
            'degradation_events': self.degradation_events,
            'baseline_avg': self.baseline_avg,
            'total_transactions': batch_num * self.batch_size,
            'total_elapsed_time': sum(self.batch_times),
            'avg_batch_time': sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        }
        
        checkpoint_path = os.path.join(output_folder, f'checkpoint_{batch_num}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def save_final_metrics(self, output_folder: str):
        """Save final metrics and summary."""
        total_batches = len(self.batch_numbers)
        total_transactions = total_batches * self.batch_size
        total_elapsed = sum(self.batch_times)
        avg_batch_time = total_elapsed / total_batches if total_batches > 0 else 0
        
        final_metrics = {
            'test_name': self.api_name,
            'total_batches': total_batches,
            'total_transactions': total_transactions,
            'total_elapsed_time': total_elapsed,
            'avg_batch_time': avg_batch_time,
            'baseline_avg': self.baseline_avg,
            'batch_numbers': self.batch_numbers,
            'batch_times': self.batch_times,
            'status_codes': self.status_codes,
            'errors': self.errors,
            'degradation_events': self.degradation_events
        }
        
        metrics_path = os.path.join(output_folder, 'stress_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        summary_path = os.path.join(output_folder, 'final_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"STRESS TEST SUMMARY: {self.api_name}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total Batches Processed: {total_batches}\n")
            f.write(f"Total Transactions: {total_transactions:,}\n")
            f.write(f"Total Elapsed Time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)\n")
            f.write(f"Average Batch Time: {avg_batch_time:.2f}s\n")
            f.write(f"Baseline Average (first 10 batches): {self.baseline_avg:.2f}s\n\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"DEGRADATION EVENTS\n")
            f.write(f"{'='*80}\n\n")
            
            if self.degradation_events:
                for event in self.degradation_events:
                    f.write(f"Batch {event['batch_num']} ({event['total_transactions']:,} transactions):\n")
                    f.write(f"  Severity: {event['severity']}\n")
                    f.write(f"  Batch Time: {event['batch_time']:.2f}s\n")
                    f.write(f"  Degradation Factor: {event['degradation_factor']:.1f}x\n")
                    f.write(f"  Baseline: {event['baseline_avg']:.2f}s\n\n")
            else:
                f.write("No degradation detected.\n\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"ERRORS\n")
            f.write(f"{'='*80}\n\n")
            
            if self.errors:
                for error in self.errors:
                    f.write(f"{error}\n")
            else:
                f.write("No errors occurred.\n")
        
        print(f"\nFinal metrics saved: {metrics_path}")
        print(f"Summary saved: {summary_path}")
    
    def run_stress_test(self, output_folder: str, checkpoint_every: int = 100):
        """Run stress test until manual stop or failure."""
        print(f"\n{'='*80}")
        print(f"STARTING STRESS TEST: {self.api_name}")
        print(f"{'='*80}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Press Ctrl+C to stop\n")
        
        os.makedirs(output_folder, exist_ok=True)
        
        batch_num = 0
        last_print_time = time.time()
        print_interval = 300
        
        def signal_handler(sig, frame):
            print("\n\nReceived stop signal. Saving final metrics...")
            self.running = False
        
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
                
                api_time, status_code, error = self.send_batch_request(descriptions, memos)
                
                self.batch_numbers.append(batch_num)
                self.batch_times.append(api_time)
                self.status_codes.append(status_code)
                
                if error:
                    error_msg = f"Batch {batch_num}: {error}"
                    self.errors.append(error_msg)
                    print(f"\n❌ ERROR at batch {batch_num}: {error}")
                    
                    if status_code == 0:
                        print("\n💥 API FAILED - Stopping test")
                        break
                
                self.detect_degradation(batch_num, api_time)
                
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    elapsed = sum(self.batch_times)
                    avg_time = sum(self.batch_times) / len(self.batch_times)
                    print(f"[{batch_num:6d} batches | {batch_num * self.batch_size:10,} txns | {elapsed/60:6.1f} min | Avg: {avg_time:.2f}s]")
                    last_print_time = current_time
                
                if batch_num % checkpoint_every == 0:
                    self.save_checkpoint(batch_num, output_folder)
                    print(f"✓ Checkpoint {batch_num} saved")
        
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {str(e)}")
            self.errors.append(f"Fatal error at batch {batch_num}: {str(e)}")
        
        finally:
            print(f"\n\n{'='*80}")
            print("STRESS TEST COMPLETED")
            print(f"{'='*80}")
            print(f"Total Batches: {batch_num:,}")
            print(f"Total Transactions: {batch_num * self.batch_size:,}")
            print(f"Total Time: {sum(self.batch_times):.2f}s ({sum(self.batch_times)/60:.2f} min)")
            print(f"Degradation Events: {len(self.degradation_events)}")
            print(f"Errors: {len(self.errors)}")
            
            self.save_final_metrics(output_folder)


def main():
    
    API_URL = 'PLACEHOLDER_API_URL'
    API_NAME = 'stress_test_api'
    BATCH_SIZE = 50
    CHECKPOINT_EVERY = 100
    
    print("="*80)
    print("API STRESS TEST")
    print("="*80)
    print(f"\nAPI URL: {API_URL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"\nThis will run until:")
    print("  - You press Ctrl+C")
    print("  - API fails/crashes")
    print("  - Critical error occurs")
    
    input("\nPress Enter to start...")
    
    stress_tester = StressTestRunner(API_NAME, API_URL, BATCH_SIZE)
    stress_tester.run_stress_test('stress_test_output', CHECKPOINT_EVERY)
    
    print("\n✅ Stress test complete!")
    print("Check stress_test_output/ for results")


if __name__ == "__main__":
    main()
