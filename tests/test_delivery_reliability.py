"""Regression checks for persisted alerts and acknowledgement ordering.

Broker doubles check failure boundaries; these are not live Kafka integration tests.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.workspace_store import WorkspaceStore
from src.kafka_streaming import FraudStreamingConsumer, SimulatedStreamingBroker


class TestDeliveryReliability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'cases.sqlite3'
        self.store = WorkspaceStore(self.path)
        self.broker = SimulatedStreamingBroker()
        self.engine_patch = patch('src.kafka_streaming.FraudModelComparison')
        engine = self.engine_patch.start().return_value
        engine.predict_single.return_value = {
            'Fraud_Votes': 1, 'Models_Checked': 4, 'Prediction': 'FLAGGED FOR REVIEW',
            'Flagged_By': ['Random Forest'],
            'models': [{'Model': 'Random Forest', 'Score': 0.9}],
        }
        self.event = {'event_id': 'replay-001', 'transaction': {
            'step': 1, 'type': 'TRANSFER', 'amount': 100,
            'oldbalanceOrg': 100, 'newbalanceOrig': 0,
            'oldbalanceDest': 0, 'newbalanceDest': 0,
        }}

    def tearDown(self):
        self.engine_patch.stop()
        self.store._connection.close()
        self.tmp.cleanup()

    def consumer(self, **kwargs):
        return FraudStreamingConsumer(use_simulated=True, simulated_broker=self.broker,
                                      store=self.store, **kwargs)

    def test_restart_after_case_save_delivers_pending_alert_once(self):
        crashed = self.consumer(simulate_crash_before_commit=True)
        with self.assertRaises(RuntimeError):
            crashed.process_message(self.event)
        alert = self.store.pending_alert('replay-001')
        self.assertIsNotNone(alert)
        self.assertEqual(self.broker.topic_size(crashed.alerts_topic), 0)
        # Reopen storage to prove pending work survives a process restart.
        self.store._connection.close()
        self.store = WorkspaceStore(self.path)
        restarted = self.consumer()
        self.assertTrue(restarted.process_message(self.event)['is_duplicate'])
        delivered = self.broker.consume(restarted.alerts_topic, timeout=0.1).deserialize()
        self.assertEqual(delivered['alert_id'], alert['alert_id'])
        self.assertIsNone(self.store.pending_alert('replay-001'))
        restarted.process_message(self.event)
        self.assertEqual(self.broker.topic_size(restarted.alerts_topic), 0)

    def test_failed_delivery_does_not_commit_and_replay_retries(self):
        consumer = self.consumer()
        consumer.use_simulated = False
        consumer._live_consumer = Mock()
        record = Mock(value=self.event, offset=7)
        consumer._live_consumer.poll.return_value = {'partition': [record]}
        consumer._live_alert_producer = Mock()
        future = consumer._live_alert_producer.send.return_value
        future.get.side_effect = TimeoutError('broker did not acknowledge')
        with self.assertRaises(TimeoutError):
            consumer.consume_one()
        consumer._live_consumer.commit.assert_not_called()
        consumer._live_consumer.seek.assert_called_once_with("partition", 7)
        self.assertIsNotNone(self.store.pending_alert('replay-001'))
        future.get.side_effect = None
        order = []
        future.get.side_effect = lambda **kw: order.append('ack')
        consumer._live_consumer.commit.side_effect = lambda: order.append('commit')
        consumer.consume_one()
        self.assertEqual(order, ['ack', 'commit'])
        self.assertIsNone(self.store.pending_alert('replay-001'))

    def test_case_and_outbox_insert_roll_back_together(self):
        with self.assertRaises(TypeError):
            self.store.create_or_get_case(
                'bad-event', 'transaction', 1, 'TRANSFER', 0.9, 'HIGH', 'FLAGGED',
                alert_payload={'alert_id': 'bad', 'transaction_id': 'transaction', 'not_json': object()},
            )
        self.assertIsNone(self.store.get_case_by_event('bad-event'))

    def test_live_alert_decoder_uses_json(self):
        from src import api
        mock_consumer = Mock()
        def make_consumer(*args, **kwargs):
            decoded = kwargs['value_deserializer'](b'{"event_id":"decoded"}')
            mock_consumer.poll.return_value = {'partition': [Mock(value=decoded)]}
            return mock_consumer
        with patch.object(api, 'authenticate_request'), patch('kafka.KafkaConsumer', side_effect=make_consumer):
            response = api.get_streaming_alerts(limit=10, mode='live')
        self.assertEqual(response['alerts'], [{'event_id': 'decoded'}])
        mock_consumer.close.assert_called_once()

    def test_compose_requires_deployment_secret(self):
        compose = (Path(__file__).resolve().parents[1] / 'docker-compose.yml').read_text()
        self.assertIn('${FRAUDGUARD_API_KEY:?Set FRAUDGUARD_API_KEY}', compose)
        self.assertNotIn('fg_deployment_secret_required', compose)

    def test_commit_failure_rewinds_without_republishing_acknowledged_alert(self):
        consumer = self.consumer()
        consumer.use_simulated = False
        consumer._live_consumer = Mock()
        consumer._live_consumer.poll.return_value = {'partition': [Mock(value=self.event, offset=8)]}
        consumer._live_alert_producer = Mock()
        consumer._live_consumer.commit.side_effect = RuntimeError('commit failed')
        with self.assertRaises(RuntimeError):
            consumer.consume_one()
        consumer._live_consumer.seek.assert_called_once_with('partition', 8)
        self.assertIsNone(self.store.pending_alert('replay-001'))
        consumer._live_consumer.commit.side_effect = None
        consumer.consume_one()
        consumer._live_alert_producer.send.assert_called_once()

    def test_other_consumed_event_is_not_reported_as_request_completion(self):
        from src import api
        tx = api.TransactionInput(**self.event['transaction'])
        with patch.object(api, 'authenticate_request'), patch.object(api, 'get_store'), \
             patch.object(api, 'TransactionStreamProducer') as producer, \
             patch.object(api, 'FraudStreamingConsumer') as consumer:
            producer.return_value.produce.return_value = {'event_id': 'requested'}
            consumer.return_value.consume_one.return_value = {'event_id': 'another'}
            response = api.publish_streaming_event(tx, mode='simulation')
        self.assertEqual(response['status'], 'published_pending_consumption')
        self.assertFalse(response['verified_consumption'])
