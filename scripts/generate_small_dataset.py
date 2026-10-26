"""Generate a realistic 10-15 MB benchmark dataset matching PaySim schema (<25MB upload limit)."""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "paysim_benchmark_12mb.csv")

def generate_sample(target_rows=120000):
    np.random.seed(42)
    types = ['PAYMENT', 'CASH_OUT', 'CASH_IN', 'TRANSFER', 'DEBIT']
    type_probs = [0.35, 0.35, 0.15, 0.13, 0.02]

    steps = np.random.randint(1, 744, size=target_rows)
    tx_types = np.random.choice(types, size=target_rows, p=type_probs)
    
    amounts = np.zeros(target_rows)
    old_origs = np.zeros(target_rows)
    new_origs = np.zeros(target_rows)
    old_dests = np.zeros(target_rows)
    new_dests = np.zeros(target_rows)
    is_frauds = np.zeros(target_rows, dtype=int)
    is_flagged = np.zeros(target_rows, dtype=int)

    # 1.5% fraud rate
    fraud_indices = set(np.random.choice(target_rows, size=int(target_rows * 0.015), replace=False))

    for i in range(target_rows):
        t = tx_types[i]
        if i in fraud_indices:
            # Fraudulent pattern: TRANSFER or CASH_OUT with complete balance depletion
            t = np.random.choice(['TRANSFER', 'CASH_OUT'])
            tx_types[i] = t
            is_frauds[i] = 1
            amt = float(np.random.uniform(50000, 1500000))
            amounts[i] = round(amt, 2)
            old_origs[i] = amounts[i]
            new_origs[i] = 0.0  # complete depletion
            old_dests[i] = 0.0
            new_dests[i] = 0.0 if t == 'TRANSFER' else round(amt, 2)
            if amt > 200000:
                is_flagged[i] = 1 if np.random.rand() > 0.4 else 0
        else:
            # Benign pattern
            is_frauds[i] = 0
            if t == 'PAYMENT':
                amt = float(np.random.exponential(scale=65.0) + 5.0)
                amounts[i] = round(amt, 2)
                old = float(np.random.uniform(amt, amt * 10 + 500))
                old_origs[i] = round(old, 2)
                new_origs[i] = round(old - amt, 2)
                old_dests[i] = round(float(np.random.uniform(0, 5000)), 2)
                new_dests[i] = round(old_dests[i] + amt, 2)
            elif t == 'CASH_OUT':
                amt = float(np.random.exponential(scale=20000.0) + 100.0)
                amounts[i] = round(amt, 2)
                old = float(np.random.uniform(amt, amt * 5 + 1000))
                old_origs[i] = round(old, 2)
                new_origs[i] = round(max(0.0, old - amt), 2)
                old_d = float(np.random.uniform(1000, 100000))
                old_dests[i] = round(old_d, 2)
                new_dests[i] = round(old_d + amt, 2)
            elif t == 'CASH_IN':
                amt = float(np.random.exponential(scale=15000.0) + 50.0)
                amounts[i] = round(amt, 2)
                old = float(np.random.uniform(0, 50000))
                old_origs[i] = round(old, 2)
                new_origs[i] = round(old + amt, 2)
                old_dests[i] = round(float(np.random.uniform(amt, amt * 3)), 2)
                new_dests[i] = round(max(0.0, old_dests[i] - amt), 2)
            elif t == 'TRANSFER':
                amt = float(np.random.exponential(scale=35000.0) + 500.0)
                amounts[i] = round(amt, 2)
                old = float(np.random.uniform(amt, amt * 4 + 2000))
                old_origs[i] = round(old, 2)
                new_origs[i] = round(old - amt, 2)
                old_d = float(np.random.uniform(500, 50000))
                old_dests[i] = round(old_d, 2)
                new_dests[i] = round(old_d + amt, 2)
            else:  # DEBIT
                amt = float(np.random.uniform(10, 2500))
                amounts[i] = round(amt, 2)
                old = float(np.random.uniform(amt, amt * 5))
                old_origs[i] = round(old, 2)
                new_origs[i] = round(old - amt, 2)
                old_d = float(np.random.uniform(0, 20000))
                old_dests[i] = round(old_d, 2)
                new_dests[i] = round(old_d + amt, 2)

    df = pd.DataFrame({
        'step': steps,
        'type': tx_types,
        'amount': amounts,
        'nameOrig': [f"C{100000000 + idx}" for idx in range(target_rows)],
        'oldbalanceOrg': old_origs,
        'newbalanceOrig': new_origs,
        'nameDest': [f"M{200000000 + idx}" if t == 'PAYMENT' else f"C{300000000 + idx}" for idx, t in enumerate(tx_types)],
        'oldbalanceDest': old_dests,
        'newbalanceDest': new_dests,
        'isFraud': is_frauds,
        'isFlaggedFraud': is_flagged
    })

    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"Generated {OUTPUT_CSV}: {len(df):,} rows, {size_mb:.2f} MB (within 25 MB limit)")

if __name__ == "__main__":
    generate_sample()
