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


if __name__ == '__main__':
    unittest.main()
