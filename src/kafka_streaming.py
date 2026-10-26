"""Real-Time Kafka Streaming Pipeline for Financial Fraud Detection.

This module provides production-grade streaming ingestion and fraud scoring:
- Supports real Apache Kafka brokers via `kafka-python` or `confluent-kafka` when available.
- Features a zero-dependency, thread-safe `SimulatedStreamingBroker` for standalone
  academic demonstration, unit tests, and air-gapped environments without external servers.
- TransactionStreamProducer: Streams transactions as JSON events with schema validation.
- FraudStreamingConsumer: Performs real-time inference using FraudModelComparison (or single
  predictors), computes balance forensic factors, measures processing latency (ms),
  and emits structured alerts to the fraud alerts topic.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import config

try:
    from model_comparison import FraudModelComparison
    from predictor import FraudPredictor
    from workspace_store import WorkspaceStore
except ImportError:
    from .model_comparison import FraudModelComparison
    from .predictor import FraudPredictor
    from .workspace_store import WorkspaceStore

logger = logging.getLogger('FraudGuard.Streaming')


# =====================================================================
# IN-MEMORY SIMULATED KAFKA BROKER (ZERO DEPENDENCY)
# =====================================================================

class StreamMessage:
    """Represents a serialized Kafka-style message with metadata."""

    def __init__(self, topic: str, key: str | None, value: bytes, headers: dict | None = None,
                 partition: int = 0, offset: int = 0):
        self.topic = topic
        self.key = key
        self.value = value
        self.headers = headers or {}
        self.partition = partition
        self.offset = offset
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def deserialize(self) -> dict:
        return json.loads(self.value.decode('utf-8'))


class SimulatedStreamingBroker:
    """Thread-safe in-memory message broker simulating Kafka topic semantics."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._topics: dict[str, queue.Queue[StreamMessage]] = {}
        self._topic_locks: dict[str, threading.Lock] = {}
        self._offsets: dict[str, int] = {}
        self._message_history: dict[str, list[StreamMessage]] = {}
        self._global_lock = threading.Lock()

    @classmethod
    def get_shared_broker(cls) -> SimulatedStreamingBroker:
        """Singleton accessor for shared in-process streaming simulations."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_topic(self, topic: str):
        with self._global_lock:
            if topic not in self._topics:
                self._topics[topic] = queue.Queue()
                self._topic_locks[topic] = threading.Lock()
                self._offsets[topic] = 0
                self._message_history[topic] = []

    def publish(self, topic: str, key: str | None, value: bytes, headers: dict | None = None) -> StreamMessage:
        """Publish a message to a topic."""
        self._ensure_topic(topic)
        with self._topic_locks[topic]:
            offset = self._offsets[topic]
            self._offsets[topic] += 1
            msg = StreamMessage(topic=topic, key=key, value=value, headers=headers, offset=offset)
            self._topics[topic].put(msg)
            self._message_history[topic].append(msg)
            return msg

    def consume(self, topic: str, timeout: float = 0.5) -> StreamMessage | None:
        """Consume a message from a topic."""
        self._ensure_topic(topic)
        try:
            return self._topics[topic].get(timeout=timeout)
        except queue.Empty:
            return None

    def topic_size(self, topic: str) -> int:
        """Return the current queue backlog size for a topic."""
        self._ensure_topic(topic)
        return self._topics[topic].qsize()

    def total_published(self, topic: str) -> int:
        """Return the total number of messages ever published to a topic."""
        self._ensure_topic(topic)
        return len(self._message_history[topic])

    def get_history(self, topic: str, limit: int = 50) -> list[StreamMessage]:
        """Inspect the most recent messages published to a topic."""
        self._ensure_topic(topic)
        with self._topic_locks[topic]:
            return list(self._message_history[topic][-limit:])

    def clear(self, topic: str | None = None):
        """Reset broker queues and histories."""
        with self._global_lock:
            if topic:
                if topic in self._topics:
                    self._topics[topic] = queue.Queue()
                    self._message_history[topic] = []
                    self._offsets[topic] = 0
            else:
                self._topics.clear()
                self._topic_locks.clear()
                self._offsets.clear()
                self._message_history.clear()


# =====================================================================
# TRANSACTION PRODUCER
# =====================================================================

class TransactionStreamProducer:
    """Produces JSON transaction events to Kafka or the simulated streaming broker."""

    def __init__(self, bootstrap_servers: str = config.KAFKA_BOOTSTRAP_SERVERS,
                 topic: str = config.KAFKA_TRANSACTIONS_TOPIC,
                 use_simulated: bool = False,
                 simulated_broker: SimulatedStreamingBroker | None = None,
                 allow_fallback: bool = True):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.use_simulated = use_simulated
        self.allow_fallback = allow_fallback
        self._broker = simulated_broker or SimulatedStreamingBroker.get_shared_broker()
        self._live_producer = None
        self._sent_count = 0

        if not self.use_simulated:
            self._init_live_producer()

    @property
    def is_live(self) -> bool:
        return not self.use_simulated and self._live_producer is not None

    def _init_live_producer(self):
        try:
            from kafka import KafkaProducer
            self._live_producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                request_timeout_ms=3000,
            )
            logger.info('Connected live KafkaProducer to %s', self.bootstrap_servers)
        except Exception as exc:
            if not self.allow_fallback:
                raise ConnectionError(
                    f"Live Kafka broker at '{self.bootstrap_servers}' is unreachable ({exc}). "
                    f"Ensure Apache Kafka is running or set use_simulated=True."
                ) from exc
            logger.warning('Live Kafka broker connection failed (%s); switching to simulated broker.', exc)
            self.use_simulated = True

    def produce(self, tx_data: dict, key: str | None = None) -> dict:
        """Validate and send a single transaction payload to the topic."""
        event_id = f'EVT-{uuid.uuid4().hex[:12].upper()}'
        now_utc = datetime.now(timezone.utc).isoformat()

        # Clean/sanitize payload for serialization
        sanitized = {}
        for k, v in tx_data.items():
            if isinstance(v, (np.integer, int)):
                sanitized[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                sanitized[k] = float(v) if np.isfinite(v) else 0.0
            else:
                sanitized[k] = str(v) if v is not None else ''

        payload = {
            'event_id': event_id,
            'timestamp_utc': now_utc,
            'transaction': sanitized,
        }

        key_str = str(key or sanitized.get('Transaction_ID') or event_id)

        if not self.use_simulated and self._live_producer is not None:
            self._live_producer.send(self.topic, key=key_str, value=payload)
        else:
            value_bytes = json.dumps(payload).encode('utf-8')
            self._broker.publish(self.topic, key=key_str, value=value_bytes,
                                 headers={'source': 'FraudGuardProducer', 'event_id': event_id})

        self._sent_count += 1
        return payload

    def stream_dataset(self, df_or_csv_path: str | pd.DataFrame,
                       rate_events_per_sec: float = 10.0,
                       max_events: int | None = None,
                       on_event_produced: Callable[[dict, int], None] | None = None) -> int:
        """Stream multiple transactions from a CSV file or DataFrame at a specified rate."""
        if isinstance(df_or_csv_path, str):
            df = pd.read_csv(df_or_csv_path)
        else:
            df = df_or_csv_path.copy()

        count = 0
        limit = max_events if max_events is not None else len(df)
        delay = (1.0 / rate_events_per_sec) if rate_events_per_sec > 0 else 0.0

        for idx, row in df.iloc[:limit].iterrows():
            tx_dict = row.to_dict()
            tx_id = tx_dict.get('Transaction_ID', f'TXN-{idx:06d}')
            payload = self.produce(tx_dict, key=str(tx_id))
            count += 1
            if on_event_produced:
                on_event_produced(payload, count)
            if delay > 0 and count < limit:
                time.sleep(delay)

        return count

    def close(self):
        if self._live_producer is not None:
            try:
                self._live_producer.flush(timeout=2)
                self._live_producer.close(timeout=2)
            except Exception:
                pass


# =====================================================================
# FRAUD STREAMING CONSUMER & DETECTOR
# =====================================================================

class FraudStreamingConsumer:
    """Consumes real-time transactions, runs ML/DL scoring, and publishes alerts."""

    def __init__(self, bootstrap_servers: str = config.KAFKA_BOOTSTRAP_SERVERS,
                 input_topic: str = config.KAFKA_TRANSACTIONS_TOPIC,
                 alerts_topic: str = config.KAFKA_ALERTS_TOPIC,
                 dlq_topic: str = config.KAFKA_DLQ_TOPIC,
                 group_id: str = config.KAFKA_CONSUMER_GROUP,
                 model_mode: str = 'comparison',
                 use_simulated: bool = False,
                 simulated_broker: SimulatedStreamingBroker | None = None,
                 models_dir: str = config.MODELS_DIR,
                 store: WorkspaceStore | None = None,
                 simulate_crash_before_commit: bool = False,
                 max_retries: int = 3,
                 allow_fallback: bool = True):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.alerts_topic = alerts_topic
        self.dlq_topic = dlq_topic
        self.group_id = group_id
        self.model_mode = model_mode
        self.use_simulated = use_simulated
        self.allow_fallback = allow_fallback
        self.models_dir = models_dir
        self.store = store
        self.simulate_crash_before_commit = simulate_crash_before_commit
        self.max_retries = max_retries
        self._broker = simulated_broker or SimulatedStreamingBroker.get_shared_broker()

        # Metrics & tracking
        self.total_consumed = 0
        self.alerts_published = 0
        self.dlq_published = 0
        self.duplicate_skips = 0
        self.legitimate_count = 0
        self.errors_count = 0
        self.total_latency_ms = 0.0
        self.alert_history: list[dict] = []
        self._history_lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._live_consumer = None
        self._live_alert_producer = None

        # Load models
        if self.model_mode == 'comparison':
            self.model_engine = FraudModelComparison(models_dir=self.models_dir)
        else:
            self.model_engine = FraudPredictor(model_name=self.model_mode, models_dir=self.models_dir)

        if not self.use_simulated:
            self._init_live_consumer()

    @property
    def is_live(self) -> bool:
        return not self.use_simulated and self._live_consumer is not None

    def _init_live_consumer(self):
        try:
            from kafka import KafkaConsumer, KafkaProducer
            self._live_consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers.split(','),
                group_id=self.group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                consumer_timeout_ms=1000,
            )
            self._live_alert_producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            logger.info('Connected live KafkaConsumer to %s on %s', self.bootstrap_servers, self.input_topic)
        except Exception as exc:
            if not self.allow_fallback:
                raise ConnectionError(
                    f"Live Kafka broker at '{self.bootstrap_servers}' is unreachable ({exc}). "
                    f"Ensure Apache Kafka is running or set use_simulated=True."
                ) from exc
            logger.warning('Live Kafka consumer connection failed (%s); switching to simulated broker.', exc)
            self.use_simulated = True

    def _extract_forensic_factors(self, tx: dict) -> list[str]:
        """Compute quick balance discrepancies and forensic observations."""
        factors = []
        amount = float(tx.get('amount', 0.0) or 0.0)
        old_orig = float(tx.get('oldbalanceOrg', 0.0) or 0.0)
        new_orig = float(tx.get('newbalanceOrig', 0.0) or 0.0)
        tx_type = str(tx.get('type', '')).upper()

        if tx_type in ('TRANSFER', 'CASH_OUT'):
            factors.append(f'High-risk transfer type: {tx_type}')
        if old_orig > 0 and new_orig == 0:
            factors.append('Sender account completely depleted')
        if old_orig > 0 and amount / old_orig > 0.85:
            factors.append(f'Amount represents {amount / old_orig * 100:.1f}% of sender balance')
        sender_diff = (old_orig - new_orig) - amount
        if abs(sender_diff) > 1.0 and tx_type in ('TRANSFER', 'CASH_OUT'):
            factors.append(f'Sender balance discrepancy: {sender_diff:,.2f} units')
        return factors or ['Standard financial balance consistency.']

    def route_to_dlq(self, raw_message: dict, reason: str, exception: Exception | None = None) -> dict:
        """Route poisonous or failing messages to the Dead-Letter Queue (DLQ)."""
        self.dlq_published += 1
        dlq_event = {
            'dlq_id': f'DLQ-{uuid.uuid4().hex[:10].upper()}',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'failed_message': raw_message,
            'failure_reason': reason,
            'error_detail': str(exception) if exception else '',
        }
        dlq_bytes = json.dumps(dlq_event).encode('utf-8')
        if not self.use_simulated and self._live_alert_producer is not None:
            self._live_alert_producer.send(self.dlq_topic, value=dlq_event)
        else:
            self._broker.publish(self.dlq_topic, key=None, value=dlq_bytes, headers={'dlq_reason': reason})
        logger.warning('Message routed to DLQ topic %s: %s', self.dlq_topic, reason)
        return dlq_event

    def process_message(self, message_dict: dict) -> dict:
        """Score a single incoming transaction event with idempotency, forensics, and alert dispatch."""
        start_time = time.perf_counter()
        tx = message_dict.get('transaction', message_dict)
        event_id = message_dict.get('event_id', f'EVT-{uuid.uuid4().hex[:8].upper()}')
        timestamp_utc = message_dict.get('timestamp_utc', datetime.now(timezone.utc).isoformat())

        # 1. Idempotency Check: prevent duplicate database effects upon Kafka offset replay
        if self.store is not None:
            existing_case = self.store.get_case_by_event(event_id)
            if existing_case is not None:
                self.duplicate_skips += 1
                logger.info('Duplicate Kafka event %s detected; skipping re-execution.', event_id)
                return {
                    'event_id': event_id,
                    'case_id': existing_case['case_id'],
                    'transaction_id': existing_case['transaction_id'],
                    'amount': existing_case['amount'],
                    'type': existing_case['txn_type'],
                    'decision': existing_case['decision'],
                    'is_flagged': existing_case['status'] != 'RESOLVED',
                    'is_duplicate': True,
                    'idempotent_action': 'SKIPPED_EXISTING_CASE',
                    'latency_ms': 0.0,
                    'forensic_factors': json.loads(existing_case.get('explanation_json', '{}')).get('forensic_factors', []),
                }

        # 2. Schema validation with bounded retry / DLQ fallback
        try:
            clean_tx = {
                'step': int(tx.get('step', 1)),
                'type': str(tx.get('type', 'PAYMENT')),
                'amount': float(tx.get('amount', 0.0)),
                'oldbalanceOrg': float(tx.get('oldbalanceOrg', 0.0)),
                'newbalanceOrig': float(tx.get('newbalanceOrig', 0.0)),
                'oldbalanceDest': float(tx.get('oldbalanceDest', 0.0)),
                'newbalanceDest': float(tx.get('newbalanceDest', 0.0)),
            }
        except (ValueError, TypeError) as exc:
            self.errors_count += 1
            return self.route_to_dlq(message_dict, f"Schema validation failure: {exc}", exc)

        if 'Transaction_ID' in tx:
            clean_tx['Transaction_ID'] = tx['Transaction_ID']

        if self.model_mode == 'comparison':
            res = self.model_engine.predict_single(clean_tx)
            is_flagged = res['Fraud_Votes'] > 0
            decision = res['Prediction']
            fraud_votes = res['Fraud_Votes']
            models_checked = res['Models_Checked']
            flagged_by = res['Flagged_By']
            model_details = res['models']
            rec_action = ('Manual analyst verification required before settlement'
                          if is_flagged else 'Standard processing')
        else:
            res = self.model_engine.predict_single(clean_tx)
            is_flagged = (res['Prediction'] == 'FRAUDULENT')
            decision = res['Prediction']
            fraud_votes = 1 if is_flagged else 0
            models_checked = 1
            flagged_by = [self.model_mode] if is_flagged else []
            model_details = [{
                'Model': self.model_mode,
                'Prediction': res['Prediction'],
                'Score': res['Fraud_Probability'],
                'Threshold': res['Decision_Threshold'],
            }]
            rec_action = res.get('Recommended_Action', 'Standard processing')

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.total_consumed += 1
        self.total_latency_ms += latency_ms

        forensic_factors = self._extract_forensic_factors(clean_tx)

        processed_record = {
            'event_id': event_id,
            'timestamp_utc': timestamp_utc,
            'transaction_id': clean_tx.get('Transaction_ID', event_id),
            'amount': clean_tx['amount'],
            'type': clean_tx['type'],
            'decision': decision,
            'is_flagged': is_flagged,
            'fraud_votes': fraud_votes,
            'models_checked': models_checked,
            'flagged_by': flagged_by,
            'model_details': model_details,
            'latency_ms': round(latency_ms, 2),
            'recommended_action': rec_action,
            'forensic_factors': forensic_factors,
            'is_duplicate': False,
        }

        # 3. Database persistence & Fraud Investigation Queue Case creation
        case_id = None
        if self.store is not None:
            threshold_val = getattr(self.model_engine, 'threshold', 0.5)
            risk_val = float(model_details[0]['Score']) if model_details else 0.0
            case_data, was_created = self.store.create_or_get_case(
                event_id=event_id,
                transaction_id=clean_tx.get('Transaction_ID', event_id),
                amount=clean_tx['amount'],
                txn_type=clean_tx['type'],
                risk_score=risk_val,
                risk_tier='CRITICAL' if is_flagged else 'LOW',
                decision=decision,
                assigned_analyst='Unassigned',
                status='OPEN' if is_flagged else 'RESOLVED',
                outcome='PENDING' if is_flagged else 'SUSPICIOUS_CLEARED',
                explanation={'forensic_factors': forensic_factors, 'flagged_by': flagged_by},
                model_version='v1.0.0-xgb-consensus',
                preprocessor_version='v1.0.0-robust',
                decision_threshold=threshold_val,
            )
            case_id = case_data.get('case_id')
            processed_record['case_id'] = case_id
            processed_record['was_created'] = was_created

            # 4. Crash simulation hook: test crash after DB write but BEFORE Kafka commit
            if self.simulate_crash_before_commit:
                logger.warning('Simulating worker process crash after database save before offset commit!')
                raise RuntimeError(f"SimulatedConsumerCrash: Process died before committing offset for event {event_id} (Case {case_id} saved).")

        if is_flagged:
            self.alerts_published += 1
            alert_id = f'ALT-{uuid.uuid4().hex[:10].upper()}'
            alert_event = {
                'alert_id': alert_id,
                **processed_record,
            }
            # Publish alert to alerts topic
            if not self.use_simulated and self._live_alert_producer is not None:
                self._live_alert_producer.send(self.alerts_topic, key=alert_event['transaction_id'], value=alert_event)
            else:
                alert_bytes = json.dumps(alert_event).encode('utf-8')
                self._broker.publish(self.alerts_topic, key=alert_event['transaction_id'],
                                     value=alert_bytes, headers={'alert_id': alert_id})

            with self._history_lock:
                self.alert_history.append(alert_event)
                if len(self.alert_history) > 200:
                    self.alert_history.pop(0)
        else:
            self.legitimate_count += 1

        return processed_record

    def consume_one(self, timeout: float = 0.5) -> dict | None:
        """Poll and process a single message from the input stream."""
        if not self.use_simulated and self._live_consumer is not None:
            raw_batch = self._live_consumer.poll(timeout_ms=int(timeout * 1000), max_records=1)
            for _, messages in raw_batch.items():
                for msg in messages:
                    return self.process_message(msg.value)
            return None
        else:
            msg = self._broker.consume(self.input_topic, timeout=timeout)
            if msg is not None:
                payload = msg.deserialize()
                return self.process_message(payload)
            return None

    def consume_batch(self, max_records: int = 50, timeout_per_message: float = 0.2) -> list[dict]:
        """Poll and process up to max_records messages."""
        results = []
        for _ in range(max_records):
            scored = self.consume_one(timeout=timeout_per_message)
            if scored is None:
                break
            results.append(scored)
        return results

    def get_metrics(self) -> dict:
        """Return current streaming throughput and alert statistics."""
        avg_latency = (self.total_latency_ms / self.total_consumed) if self.total_consumed > 0 else 0.0
        return {
            'total_consumed': self.total_consumed,
            'alerts_published': self.alerts_published,
            'legitimate_count': self.legitimate_count,
            'errors_count': self.errors_count,
            'avg_latency_ms': round(avg_latency, 2),
            'alert_rate_pct': round((self.alerts_published / self.total_consumed * 100.0), 2) if self.total_consumed > 0 else 0.0,
            'alerts_topic': self.alerts_topic,
            'input_topic': self.input_topic,
        }

    def start_background_worker(self, poll_interval: float = 0.05):
        """Start a background thread that continuously consumes and scores events."""
        if self._running:
            return
        self._running = True

        def _worker_loop():
            while self._running:
                try:
                    self.consume_one(timeout=poll_interval)
                except Exception as err:
                    self.errors_count += 1
                    logger.error('Error during streaming consumption: %s', err)
                time.sleep(poll_interval)

        self._worker_thread = threading.Thread(target=_worker_loop, daemon=True, name='FraudStreamingWorker')
        self._worker_thread.start()

    def stop(self):
        """Stop background worker and close live connections."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        if self._live_consumer is not None:
            try:
                self._live_consumer.close()
            except Exception:
                pass
        if self._live_alert_producer is not None:
            try:
                self._live_alert_producer.close()
            except Exception:
                pass


# =====================================================================
# STREAMING PIPELINE RUNNER
# =====================================================================

class FraudStreamingPipeline:
    """Convenience orchestrator combining Producer, Broker, and Consumer."""

    def __init__(self, mode: str = 'simulated',
                 bootstrap_servers: str = config.KAFKA_BOOTSTRAP_SERVERS,
                 transactions_topic: str = config.KAFKA_TRANSACTIONS_TOPIC,
                 alerts_topic: str = config.KAFKA_ALERTS_TOPIC,
                 model_mode: str = 'comparison'):
        self.mode = mode
        use_simulated = (mode.lower() == 'simulated')
        self.broker = SimulatedStreamingBroker() if use_simulated else None
        self.producer = TransactionStreamProducer(
            bootstrap_servers=bootstrap_servers,
            topic=transactions_topic,
            use_simulated=use_simulated,
            simulated_broker=self.broker,
        )
        self.consumer = FraudStreamingConsumer(
            bootstrap_servers=bootstrap_servers,
            input_topic=transactions_topic,
            alerts_topic=alerts_topic,
            model_mode=model_mode,
            use_simulated=use_simulated,
            simulated_broker=self.broker,
        )

    def run_simulation(self, sample_csv_or_df: str | pd.DataFrame,
                       num_events: int = 50,
                       rate_per_sec: float = 20.0,
                       on_scored_callback: Callable[[dict], None] | None = None) -> dict:
        """Run a synchronized streaming simulation and return comprehensive metrics."""
        self.consumer.start_background_worker(poll_interval=0.01)
        try:
            sent = self.producer.stream_dataset(
                sample_csv_or_df,
                rate_events_per_sec=rate_per_sec,
                max_events=num_events,
            )
            # Allow consumer to drain any remaining queued messages
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self.consumer.total_consumed >= sent:
                    break
                time.sleep(0.05)
        finally:
            self.consumer.stop()
            self.producer.close()

        metrics = self.consumer.get_metrics()
        metrics['events_produced'] = sent
        metrics['recent_alerts'] = list(self.consumer.alert_history)
        return metrics
