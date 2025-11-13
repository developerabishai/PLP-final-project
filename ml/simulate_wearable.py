import csv
import time
import random
from datetime import datetime

OUTFILE = "simulated_wearable.csv"

def generate_record():
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "heart_rate": float(round(random.uniform(55.0, 120.0), 2)),
        "hr_variability": float(round(random.uniform(10.0, 80.0), 2)),
        "blood_oxygen": float(round(random.uniform(90.0, 99.9), 2)),
        "steps": float(round(random.uniform(0.0, 5.0), 2)),
    }
    return record

def run_simulation(n=200):
    with open(OUTFILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","heart_rate","hr_variability","blood_oxygen","steps"])
        writer.writeheader()
        for _ in range(n):
            r = generate_record()
            writer.writerow(r)
            print("WROTE:", r)
            time.sleep(0.01)

if __name__ == "__main__":
    run_simulation(200)
