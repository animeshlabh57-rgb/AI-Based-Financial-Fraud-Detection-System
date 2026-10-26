import os
import sys
from datetime import datetime
import json
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config
import tempfile
from src.workspace_store import WorkspaceStore
from src.kafka_streaming import (
    FraudStreamingConsumer,
    FraudStreamingPipeline,
    SimulatedStreamingBroker,
    StreamMessage,
    TransactionStreamProducer,
)


class TestKafkaStreaming(unittest.TestCase):
    def setUp(self):
        self.broker = SimulatedStreamingBroker()
        self.tx_topic = 'test.transactions'
        self.alert_topic = 'test.alerts'
        self.dlq_topic = 'test.dlq'
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False)
        self.temp_db.close()
        self.store = WorkspaceStore(db_path=self.temp_db.name)

        self.legit_tx = {
            'step': 10,
            'type': 'PAYMENT',
            'amount': 250.0,
            'oldbalanceOrg': 5000.0,
            'newbalanceOrig': 4750.0,
            'oldbalanceDest': 1000.0,
            'newbalanceDest': 1250.0,
            'Transaction_ID': 'TEST-LEGIT-001',
        }

        # Known fraud pattern (sender balance depleted, large transfer)
        self.fraud_tx = {
            'step': 12,
            'type': 'TRANSFER',
            'amount': 150000.0,
            'oldbalanceOrg': 150000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'Transaction_ID': 'TEST-FRAUD-001',
        }

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.unlink(self.temp_db.name)
            except Exception:
                pass

    def test_simulated_broker_fifo_and_offsets(self):
        broker = SimulatedStreamingBroker()
        broker.publish('t1', key='k1', value=b'msg1')
        broker.publish('t1', key='k2', value=b'msg2')

        self.assertEqual(broker.topic_size('t1'), 2)
        m1 = broker.consume('t1', timeout=0.1)
        self.assertIsNotNone(m1)
        self.assertEqual(m1.key, 'k1')
        self.assertEqual(m1.value, b'msg1')
        self.assertEqual(m1.offset, 0)

        m2 = broker.consume('t1', timeout=0.1)
        self.assertIsNotNone(m2)
        self.assertEqual(m2.offset, 1)

        self.assertEqual(broker.topic_size('t1'), 0)
        self.assertIsNone(broker.consume('t1', timeout=0.05))

    def test_producer_json_sanitization_and_metadata(self):
        producer = TransactionStreamProducer(
            topic=self.tx_topic,
            use_simulated=True,
            simulated_broker=self.broker,
        )
        payload = producer.produce(self.legit_tx, key='KEY-001')
        self.assertIn('event_id', payload)
        self.assertIn('timestamp_utc', payload)
        self.assertEqual(payload['transaction']['amount'], 250.0)
        self.assertEqual(self.broker.topic_size(self.tx_topic), 1)

    def test_consumer_scores_legitimate_event_no_alert(self):
        producer = TransactionStreamProducer(
            topic=self.tx_topic,
            use_simulated=True,
            simulated_broker=self.broker,
        )
        consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='comparison',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
        )

        producer.produce(self.legit_tx)
        res = consumer.consume_one(timeout=0.2)

        self.assertIsNotNone(res)
        self.assertFalse(res['is_flagged'])
        self.assertEqual(consumer.alerts_published, 0)
        self.assertEqual(consumer.legitimate_count, 1)
        self.assertEqual(self.broker.topic_size(self.alert_topic), 0)

    def test_consumer_detects_and_publishes_fraud_alert(self):
        producer = TransactionStreamProducer(
            topic=self.tx_topic,
            use_simulated=True,
            simulated_broker=self.broker,
        )
        consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='comparison',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
        )

        producer.produce(self.fraud_tx)
        res = consumer.consume_one(timeout=0.2)

        self.assertIsNotNone(res)
        self.assertTrue(res['is_flagged'])
        self.assertGreater(res['fraud_votes'], 0)
        self.assertEqual(consumer.alerts_published, 1)
        self.assertEqual(self.broker.topic_size(self.alert_topic), 1)

        alert_msg = self.broker.consume(self.alert_topic, timeout=0.1)
        self.assertIsNotNone(alert_msg)
        alert_dict = alert_msg.deserialize()
        self.assertIn('alert_id', alert_dict)
        self.assertEqual(alert_dict['transaction_id'], 'TEST-FRAUD-001')
        self.assertGreaterEqual(len(alert_dict['forensic_factors']), 1)

    def test_consumer_single_model_mode(self):
        producer = TransactionStreamProducer(
            topic=self.tx_topic,
            use_simulated=True,
            simulated_broker=self.broker,
        )
        consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='rf',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
        )

        producer.produce(self.fraud_tx)
        res = consumer.consume_one(timeout=0.2)

        self.assertIsNotNone(res)
        self.assertEqual(res['models_checked'], 1)
        self.assertIn('latency_ms', res)

    def test_streaming_pipeline_simulation_run(self):
        sample_df = pd.DataFrame([self.legit_tx, self.fraud_tx])
        pipeline = FraudStreamingPipeline(
            mode='simulated',
            transactions_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='comparison',
        )

        metrics = pipeline.run_simulation(sample_df, num_events=2, rate_per_sec=50.0)

        self.assertEqual(metrics['events_produced'], 2)
        self.assertEqual(metrics['total_consumed'], 2)
        self.assertGreaterEqual(metrics['alerts_published'], 1)
        self.assertGreater(metrics['avg_latency_ms'], 0.0)
        self.assertEqual(metrics['errors_count'], 0)

    def test_end_to_end_broker_queue_drain_and_alert_dispatch(self):
        """Verify events actually enter the broker topic queue, are drained via consume_one, and alerts enter the alerts topic."""
        producer = TransactionStreamProducer(
            topic=self.tx_topic,
            use_simulated=True,
            simulated_broker=self.broker,
        )
        consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='comparison',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
            store=self.store,
        )

        # 1. Topic initially empty
        self.assertEqual(self.broker.topic_size(self.tx_topic), 0)
        self.assertEqual(self.broker.topic_size(self.alert_topic), 0)

        # 2. Producer enqueues event to broker
        payload = producer.produce(self.fraud_tx, key='TXN-STREAM-001')
        self.assertEqual(self.broker.topic_size(self.tx_topic), 1)

        # 3. Consumer polls and drains the queue
        processed = consumer.consume_one(timeout=0.5)
        self.assertIsNotNone(processed)
        self.assertEqual(processed['event_id'], payload['event_id'])
        self.assertTrue(processed['is_flagged'])
        self.assertEqual(self.broker.topic_size(self.tx_topic), 0)

        # 4. Verified alert is sitting in alerts topic queue
        self.assertEqual(self.broker.topic_size(self.alert_topic), 1)
        alert_msg = self.broker.consume(self.alert_topic, timeout=0.1)
        self.assertIsNotNone(alert_msg)
        alert_data = alert_msg.deserialize()
        self.assertEqual(alert_data['event_id'], payload['event_id'])
        self.assertIn('alert_id', alert_data)
        self.assertEqual(self.broker.topic_size(self.alert_topic), 0)

    def test_consumer_idempotency_and_duplicate_event_handling(self):
        """Verify replaying an event with identical event_id is safely detected and skipped without duplicate DB records."""
        producer = TransactionStreamProducer(
            topic=self.tx_topic,
            use_simulated=True,
            simulated_broker=self.broker,
        )
        consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='comparison',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
            store=self.store,
        )

        fixed_event_id = "EVT-IDEMPOTENT-TEST-999"
        message_body = {
            'event_id': fixed_event_id,
            'timestamp_utc': datetime.now().isoformat(),
            'transaction': self.fraud_tx,
        }

        # First consumption: processed normally, creates DB case
        self.broker.publish(self.tx_topic, key='k1', value=json.dumps(message_body).encode('utf-8'))
        first_result = consumer.consume_one(timeout=0.5)
        self.assertIsNotNone(first_result)
        self.assertFalse(first_result.get('is_duplicate', False))
        first_case_id = first_result.get('case_id')
        self.assertIsNotNone(first_case_id)

        # Second consumption of the same event (simulating Kafka replay / duplicate delivery)
        self.broker.publish(self.tx_topic, key='k1', value=json.dumps(message_body).encode('utf-8'))
        second_result = consumer.consume_one(timeout=0.5)
        self.assertIsNotNone(second_result)
        self.assertTrue(second_result.get('is_duplicate', False))
        self.assertEqual(second_result.get('idempotent_action'), 'SKIPPED_EXISTING_CASE')
        self.assertEqual(second_result.get('case_id'), first_case_id)
        self.assertEqual(consumer.duplicate_skips, 1)

    def test_consumer_crash_recovery_before_offset_commit(self):
        """Verify consumer crash recovery: exception raised after DB save, then fresh consumer recovers cleanly."""
        crash_consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='comparison',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
            store=self.store,
            simulate_crash_before_commit=True,
        )

        test_event = {
            'event_id': 'EVT-CRASH-TEST-001',
            'timestamp_utc': datetime.now().isoformat(),
            'transaction': self.fraud_tx,
        }

        # Worker crashes after DB write
        with self.assertRaises(RuntimeError) as ctx:
            crash_consumer.process_message(test_event)
        self.assertIn('SimulatedConsumerCrash', str(ctx.exception))

        # Fresh worker instance starts after crash and reprocesses
        recovered_consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            model_mode='comparison',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
            store=self.store,
            simulate_crash_before_commit=False,
        )
        # Verify no alert was published prior to crash
        self.assertEqual(self.broker.topic_size(self.alert_topic), 0)

        recovery_result = recovered_consumer.process_message(test_event)
        self.assertTrue(recovery_result['is_duplicate'])
        self.assertTrue(recovery_result['alert_delivered_on_recovery'])
        self.assertEqual(recovery_result['idempotent_action'], 'DELIVERED_PENDING_ALERT_ON_REPLAY')
        # Verify recovery delivered the alert to the alert topic
        self.assertEqual(self.broker.topic_size(self.alert_topic), 1)

        # Replaying again after successful delivery avoids duplicate alert
        second_recovery = recovered_consumer.process_message(test_event)
        self.assertEqual(second_recovery['idempotent_action'], 'SKIPPED_EXISTING_CASE')
        self.assertEqual(self.broker.topic_size(self.alert_topic), 1)

    def test_dead_letter_queue_and_malformed_event_handling(self):
        """Verify malformed messages are safely routed to DLQ without halting stream consumption."""
        consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            dlq_topic=self.dlq_topic,
            model_mode='comparison',
            use_simulated=True,
            simulated_broker=self.broker,
            models_dir=config.MODELS_DIR,
            store=self.store,
        )

        # Publish a poisonous message with non-convertible data
        poison_event = {
            'event_id': 'EVT-POISON-001',
            'transaction': {
                'type': 'CASH_OUT',
                'amount': 'CORRUPT_NOT_A_FLOAT',
                'oldbalanceOrg': 'CORRUPT',
            }
        }
        self.broker.publish(self.tx_topic, key='err', value=json.dumps(poison_event).encode('utf-8'))

        res = consumer.consume_one(timeout=0.5)
        self.assertIsNotNone(res)
        self.assertIn('dlq_id', res)
        self.assertEqual(consumer.dlq_published, 1)
        self.assertEqual(self.broker.topic_size(self.dlq_topic), 1)

    def test_live_consumer_manual_commit_and_crash_recovery_mechanics(self):
        """Verify that live Kafka consumer enforces manual offset commits (enable_auto_commit=False) and crash recovery."""
        class MockKafkaRecord:
            def __init__(self, value, offset):
                self.value = value
                self.offset = offset

        class MockKafkaConsumer:
            def __init__(self, messages):
                self.messages = list(messages)
                self.committed_offsets = []
                self.seek_calls = []

            def poll(self, timeout_ms=1000, max_records=1):
                if not self.messages:
                    return {}
                msg = self.messages.pop(0)
                return {'topic': [msg]}

            def commit(self):
                self.committed_offsets.append(True)

            def seek(self, partition, offset):
                self.seek_calls.append((partition, offset))

        event_payload = {
            'event_id': 'EVT-LIVE-CRASH-001',
            'timestamp_utc': datetime.now().isoformat(),
            'transaction': self.fraud_tx,
        }

        # 1. Normal execution: consumer processes and manually commits offset
        consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            use_simulated=True,
            models_dir=config.MODELS_DIR,
            store=self.store,
        )
        mock_c = MockKafkaConsumer([MockKafkaRecord(event_payload, offset=0)])
        consumer._live_consumer = mock_c
        consumer.use_simulated = False

        res = consumer.consume_one(timeout=0.5)
        self.assertIsNotNone(res)
        self.assertEqual(len(mock_c.committed_offsets), 1)  # Verified manual commit executed!
        case_id = res['case_id']

        # 2. Crash scenario: consumer dies after DB write before commit
        crash_event_payload = {
            'event_id': 'EVT-LIVE-CRASH-002',
            'timestamp_utc': datetime.now().isoformat(),
            'transaction': self.fraud_tx,
        }
        crash_consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            use_simulated=True,
            models_dir=config.MODELS_DIR,
            store=self.store,
            simulate_crash_before_commit=True,
        )
        mock_crash_c = MockKafkaConsumer([MockKafkaRecord(crash_event_payload, offset=1)])
        crash_consumer._live_consumer = mock_crash_c
        crash_consumer.use_simulated = False

        with self.assertRaises(RuntimeError):
            crash_consumer.consume_one(timeout=0.5)
        # Verified: commit was NOT executed due to crash!
        self.assertEqual(len(mock_crash_c.committed_offsets), 0)
        self.assertEqual(mock_crash_c.seek_calls, [('topic', 1)])

        # 3. Restarted consumer re-reads uncommitted offset and handles idempotently
        recovered_consumer = FraudStreamingConsumer(
            input_topic=self.tx_topic,
            alerts_topic=self.alert_topic,
            use_simulated=True,
            models_dir=config.MODELS_DIR,
            store=self.store,
            simulate_crash_before_commit=False,
        )
        mock_recovered_c = MockKafkaConsumer([MockKafkaRecord(crash_event_payload, offset=1)])
        recovered_consumer._live_consumer = mock_recovered_c
        recovered_consumer.use_simulated = False

        recovery_res = recovered_consumer.consume_one(timeout=0.5)
        self.assertIsNotNone(recovery_res)
        self.assertTrue(recovery_res['is_duplicate'])
        self.assertTrue(recovery_res['alert_delivered_on_recovery'])
        self.assertEqual(recovery_res['idempotent_action'], 'DELIVERED_PENDING_ALERT_ON_REPLAY')
        # Verified: commit succeeded on replay!
        self.assertEqual(len(mock_recovered_c.committed_offsets), 1)


if __name__ == '__main__':
    unittest.main()
