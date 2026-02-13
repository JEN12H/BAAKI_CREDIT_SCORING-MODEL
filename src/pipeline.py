import os
import sys
import time

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generation.snap import run_snap_pipeline

# Configuration (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMERS_FILE = os.path.join(PROJECT_ROOT, "data", "customers.csv")
BEHAVIOR_FILE = os.path.join(PROJECT_ROOT, "data", "credit_behavior_monthly.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "model_snapshots.csv")

def get_last_modified(file_path):
    if not os.path.exists(file_path):
        return 0
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0

def main():
    print("Starting Data Pipeline...")
    
    # Check if files exist
    if not os.path.exists(CUSTOMERS_FILE) or not os.path.exists(BEHAVIOR_FILE):
        print(f"❌ Error: Required input files not found in 'data/' folder.")
        print(f"Looked for: {CUSTOMERS_FILE} and {BEHAVIOR_FILE}")
        return

    print(f"Monitoring: {os.path.basename(CUSTOMERS_FILE)} and {os.path.basename(BEHAVIOR_FILE)}")

    # Run initially
    print("-" * 30)
    print("Initial calculation...")
    run_snap_pipeline(CUSTOMERS_FILE, BEHAVIOR_FILE, OUTPUT_FILE)
    print("-" * 30)

    # Watch Mode
    if "--watch" in sys.argv:
        print(f"Watch mode active. Monitoring for changes...")
        last_mtime_customers = get_last_modified(CUSTOMERS_FILE)
        last_mtime_behavior = get_last_modified(BEHAVIOR_FILE)

        try:
            while True:
                time.sleep(2)  # Check every 2 seconds
                
                curr_mtime_customers = get_last_modified(CUSTOMERS_FILE)
                curr_mtime_behavior = get_last_modified(BEHAVIOR_FILE)

                if curr_mtime_customers > last_mtime_customers or curr_mtime_behavior > last_mtime_behavior:
                    print("\nChange detected! Updating snapshots...")
                    success = run_snap_pipeline(CUSTOMERS_FILE, BEHAVIOR_FILE, OUTPUT_FILE)
                    if success:
                        last_mtime_customers = curr_mtime_customers
                        last_mtime_behavior = curr_mtime_behavior
                    print("-" * 30)
                    print("Listening for changes...")
        except KeyboardInterrupt:
            print("\nStopping watch mode.")
    else:
        print("\nDone. To keep monitoring for changes, run: python src/pipeline.py --watch")

if __name__ == "__main__":
    main()
