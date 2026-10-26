"""
Financial Transaction Dataset Generator (Realistic Benchmark)
Generates a realistic benchmark financial fraud dataset simulating transaction patterns,
balance shifts, and known fraud typologies (e.g., account takeover, high-velocity transfers,
empty-balance cash-outs) with natural statistical overlap and noise.
"""

import numpy as np
import pandas as pd
import os

def generate_financial_dataset(n_samples=15000, fraud_ratio=0.015, random_state=42):
    """
    Generates a realistic benchmark financial dataset with natural feature overlap,
    simulating genuine financial transaction dynamics and realistic false positive/negative trade-offs.
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    type_probs_legit = [0.42, 0.18, 0.24, 0.06, 0.10]
    
    # 1. Generate Legitimate Transactions
    legit_steps = np.random.randint(1, 744, size=n_legit)
    legit_types = np.random.choice(transaction_types, size=n_legit, p=type_probs_legit)
    
    legit_amounts = np.zeros(n_legit)
    for i, t in enumerate(legit_types):
        if t == 'PAYMENT':
            legit_amounts[i] = np.random.exponential(scale=95) + 5
        elif t == 'TRANSFER':
            legit_amounts[i] = np.random.exponential(scale=3200) + 50
        elif t == 'CASH_OUT':
            legit_amounts[i] = np.random.exponential(scale=2400) + 20
        elif t == 'DEBIT':
            legit_amounts[i] = np.random.exponential(scale=180) + 10
        elif t == 'CASH_IN':
            legit_amounts[i] = np.random.exponential(scale=1600) + 30
            
    legit_old_orig = np.random.exponential(scale=12000, size=n_legit) + legit_amounts
    
    # Legitimate origin balance post-transaction (with occasional overdraft/zero balances)
    legit_drain_noise = np.random.normal(1.0, 0.05, size=n_legit)
    legit_new_orig = np.maximum(0.0, legit_old_orig - (legit_amounts * legit_drain_noise))
    
    # Destination balances for legitimate transactions
    legit_old_dest = np.random.exponential(scale=22000, size=n_legit)
    # Some legitimate transactions go to new/merchant accounts with 0 initial balance
    zero_dest_mask = np.random.binomial(1, 0.18, size=n_legit)
    legit_old_dest = np.where(zero_dest_mask == 1, 0.0, legit_old_dest)
    legit_dest_noise = np.random.normal(0.0, 45.0, size=n_legit)
    legit_new_dest = np.maximum(0.0, legit_old_dest + legit_amounts + legit_dest_noise)
    
    legit_is_fraud = np.zeros(n_legit, dtype=int)
    legit_is_flagged = np.where(legit_amounts > 200000, 1, 0)
    
    # 2. Generate Fraudulent Transactions (Natural Overlap & Subtle Adversarial Noise)
    fraud_steps = np.random.randint(1, 744, size=n_fraud)
    # Fraud predominantly targets TRANSFER and CASH_OUT, with low-value PAYMENT probes
    fraud_types = np.random.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT'], size=n_fraud, p=[0.52, 0.40, 0.08])
    
    fraud_amounts = np.zeros(n_fraud)
    for i, t in enumerate(fraud_types):
        if t == 'PAYMENT':
            # Micro-probing fraud
            fraud_amounts[i] = np.random.exponential(scale=350) + 15
        elif t == 'TRANSFER':
            # High-value account takeover
            fraud_amounts[i] = np.random.exponential(scale=85000) + 5000
        else:
            # Medium-to-high cash-out
            fraud_amounts[i] = np.random.exponential(scale=45000) + 2000

    fraud_old_orig = fraud_amounts + np.random.exponential(scale=8000, size=n_fraud)
    
    # In fraud, origin balance is often depleted, but fraudsters sometimes leave residue to evade rules
    orig_drain_pct = np.random.uniform(0.65, 1.0, size=n_fraud)
    fraud_new_orig = np.maximum(0.0, fraud_old_orig - (fraud_amounts * orig_drain_pct))
    
    # Destination balances for fraud: some go to established mule accounts, some to fresh mules
    fraud_old_dest = np.random.choice([0.0, 1500.0, 12000.0, 45000.0], size=n_fraud, p=[0.50, 0.25, 0.15, 0.10])
    # Mule cashing out immediately vs keeping balance
    mule_drained = np.random.binomial(1, 0.30, size=n_fraud)
    fraud_dest_noise = np.random.normal(0.0, 60.0, size=n_fraud)
    fraud_new_dest = np.where(mule_drained == 1, 0.0, fraud_old_dest + fraud_amounts + fraud_dest_noise)
    
    fraud_is_fraud = np.ones(n_fraud, dtype=int)
    fraud_is_flagged = np.where(fraud_amounts > 200000, 1, 0)
    
    # 3. Assemble and Shuffle Dataset
    df_legit = pd.DataFrame({
        'step': legit_steps,
        'type': legit_types,
        'amount': np.round(legit_amounts, 2),
        'oldbalanceOrg': np.round(legit_old_orig, 2),
        'newbalanceOrig': np.round(legit_new_orig, 2),
        'oldbalanceDest': np.round(legit_old_dest, 2),
        'newbalanceDest': np.round(legit_new_dest, 2),
        'isFlaggedFraud': legit_is_flagged,
        'isFraud': legit_is_fraud
    })
    
    df_fraud = pd.DataFrame({
        'step': fraud_steps,
        'type': fraud_types,
        'amount': np.round(fraud_amounts, 2),
        'oldbalanceOrg': np.round(fraud_old_orig, 2),
        'newbalanceOrig': np.round(fraud_new_orig, 2),
        'oldbalanceDest': np.round(fraud_old_dest, 2),
        'newbalanceDest': np.round(fraud_new_dest, 2),
        'isFlaggedFraud': fraud_is_flagged,
        'isFraud': fraud_is_fraud
    })
    
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def main():
    os.makedirs('data', exist_ok=True)
    print("Generating realistic benchmark financial transactions dataset...")
    df = generate_financial_dataset(n_samples=15000, fraud_ratio=0.015, random_state=42)
    output_path = os.path.join('data', 'transactions.csv')
    df.to_csv(output_path, index=False)
    print(f"Benchmark dataset successfully saved at: '{output_path}'")
    print(f"Dataset Shape: {df.shape} (15,000 transactions, ~1.5% fraud)")
    
    # Generate a realistic sample CSV for dashboard batch testing
    sample_test = df.sample(n=30, random_state=101).copy()
    sample_path = os.path.join('data', 'sample_test.csv')
    sample_test.to_csv(sample_path, index=False)
    print(f"Sample test CSV created at: '{sample_path}'")

if __name__ == '__main__':
    main()
