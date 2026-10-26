"""Real-Time Kafka Streaming Demo CLI for FraudGuard AI.

Usage:
    python scripts/run_kafka_streaming.py [--events 30] [--rate 10] [--mode simulated] [--model comparison]
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'src'))

import config
from kafka_streaming import FraudStreamingPipeline, SimulatedStreamingBroker


def parse_args():
    parser = argparse.ArgumentParser(description='FraudGuard AI Real-Time Kafka Streaming Demo')
    parser.add_argument('--events', type=int, default=30, help='Total transaction events to stream')
    parser.add_argument('--rate', type=float, default=10.0, help='Events produced per second')
    parser.add_argument('--mode', choices=['simulated', 'live'], default='simulated',
                        help='Streaming engine: "simulated" (zero-dependency) or "live" (Apache Kafka)')
    parser.add_argument('--model', choices=['comparison', 'rf', 'xgb', 'dnn', 'ae'], default='comparison',
                        help='Model engine: comparison (all 4 models) or single model')
    parser.add_argument('--broker', default=config.KAFKA_BOOTSTRAP_SERVERS, help='Kafka bootstrap servers')
    parser.add_argument('--dataset', default=config.SAMPLE_TEST_PATH, help='Input dataset path')
    return parser.parse_args()


def main():
    args = parse_args()
    print('=' * 75)
    print(' FRAUDGUARD AI - REAL-TIME TRANSACTION STREAMING PIPELINE')
    print('=' * 75)
    print(f' Engine Mode:       {args.mode.upper()}')
    print(f' Bootstrap Servers: {args.broker}')
    print(f' Input Topic:       {config.KAFKA_TRANSACTIONS_TOPIC}')
    print(f' Alerts Topic:      {config.KAFKA_ALERTS_TOPIC}')
    print(f' Scoring Engine:    {args.model.upper()}')
    print(f' Stream Target:     {args.events} events @ {args.rate} events/sec')
    print(f' Source Dataset:    {args.dataset}')
    print('=' * 75)

    pipeline = FraudStreamingPipeline(
        mode=args.mode,
        bootstrap_servers=args.broker,
        transactions_topic=config.KAFKA_TRANSACTIONS_TOPIC,
        alerts_topic=config.KAFKA_ALERTS_TOPIC,
        model_mode=args.model,
    )

    print('\n[1/3] Starting streaming consumer in background...')
    pipeline.consumer.start_background_worker(poll_interval=0.01)

    print(f'[2/3] Ingesting and publishing {args.events} events to {config.KAFKA_TRANSACTIONS_TOPIC}...\n')

    alerts_seen = []

    def log_event(payload, idx):
        tx = payload['transaction']
        tx_id = tx.get('Transaction_ID', f'TX-{idx}')
        amt = float(tx.get('amount', 0.0))
        tx_type = tx.get('type', 'UNKNOWN')
        print(f'  --> [PRODUCED #{idx:02d}] {tx_id} | {tx_type:<8} | Amount: {amt:11,.2f} | Time: {payload["timestamp_utc"][11:19]}')

    pipeline.producer.stream_dataset(
        args.dataset,
        rate_events_per_sec=args.rate,
        max_events=args.events,
        on_event_produced=log_event,
    )

    # Wait for consumer to process all messages
    print('\n[3/3] Awaiting consumer processing & alert generation...')
    start_drain = time.time()
    while pipeline.consumer.total_consumed < args.events and (time.time() - start_drain) < 4.0:
        time.sleep(0.05)

    pipeline.consumer.stop()
    pipeline.producer.close()

    metrics = pipeline.consumer.get_metrics()
    alerts = pipeline.consumer.alert_history

    print('\n' + '=' * 75)
    print(' STREAMING RUN METRICS & SUMMARY')
    print('=' * 75)
    print(f' Total Transactions Consumed: {metrics["total_consumed"]}')
    print(f' Legitimate Transactions:     {metrics["legitimate_count"]}')
    print(f' Fraud Alerts Emitted:        {metrics["alerts_published"]} ({metrics["alert_rate_pct"]}%)')
    print(f' Average Scoring Latency:     {metrics["avg_latency_ms"]} ms / transaction')
    print(f' Processing Errors:           {metrics["errors_count"]}')
    print('=' * 75)

    if alerts:
        print(f'\n[!] FLAGGED ALERTS PUBLISHED TO {config.KAFKA_ALERTS_TOPIC} ({len(alerts)} items):')
        for i, alt in enumerate(alerts, 1):
            print(f'\n  Alert #{i}: {alt["alert_id"]} for {alt["transaction_id"]}')
            print(f'    Type: {alt["type"]} | Amount: {alt["amount"]:,.2f}')
            print(f'    Consensus: {alt["fraud_votes"]}/{alt["models_checked"]} models flagged ({", ".join(alt["flagged_by"])})')
            print(f'    Action: {alt["recommended_action"]}')
            print(f'    Forensics: {", ".join(alt["forensic_factors"][:2])}')
            print(f'    Latency: {alt["latency_ms"]} ms')
    else:
        print('\n[*] No fraudulent patterns detected in this stream batch.')

    print('\n[Done] Pipeline streaming cycle completed successfully.')


if __name__ == '__main__':
    main()
