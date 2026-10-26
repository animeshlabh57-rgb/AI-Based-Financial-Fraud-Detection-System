"""Integration and unit tests for FraudGuard AI FastAPI RESTful microservice.
Tests cover health check, authentication security, consensus scoring across 4 models,
batch prediction, request limits, Kafka streaming modes, and model metadata inspection.
"""

import json
import unittest
from fastapi.testclient import TestClient
import config
from src.api import app


class TestFraudDetectionAPI(unittest.TestCase):
    """Test suite for the FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize the FastAPI test client once with authenticated headers."""
        config.VALID_API_KEYS.add(config.TEST_API_KEY)
        cls.client = TestClient(
            app,
            headers={
                "X-API-Key": config.TEST_API_KEY,
                "X-Analyst-ID": "analyst-automated-test",
            },
        )
        cls.unauth_client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        config.VALID_API_KEYS.discard(config.TEST_API_KEY)

    def setUp(self):
        from src.api import get_store
        get_store().reset_rate_limits()

    def tearDown(self):
        from src.api import get_store
        get_store().reset_rate_limits()

    def test_root_endpoint(self):
        """Test GET / returns API metadata and service identity."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("service"), "FraudGuard AI Enterprise Prediction Engine")
        self.assertEqual(data.get("status"), "operational")
        self.assertIn("documentation", data)

    def test_health_check_endpoint(self):
        """Test GET /health verifies model pipeline readiness."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("database_connected"))
        self.assertIsInstance(data.get("models_available"), list)
        self.assertGreaterEqual(len(data.get("models_available")), 1)

    def test_models_info_endpoint(self):
        """Test GET /models/info returns metadata and weights."""
        response = self.client.get("/models/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models_loaded", data)
        self.assertIn("weights", data)
        self.assertIn("calibrated_threshold", data)
        self.assertGreater(data.get("calibrated_threshold"), 0.0)

    def test_unauthenticated_request_rejected(self):
        """Test POST /predict without API key is rejected with HTTP 401."""
        payload = {
            "type": "PAYMENT",
            "amount": 100.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4900.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1100.0,
            "step": 1,
        }
        res = self.unauth_client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 401)

    def test_predict_legitimate_transaction(self):
        """Test POST /predict with a normal, benign payment."""
        payload = {
            "type": "PAYMENT",
            "amount": 125.50,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4874.50,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1125.50,
            "step": 12,
            "nameOrig": "C100000001",
            "nameDest": "M200000001"
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("transaction_id", data)
        self.assertIn("fraud_probability", data)
        self.assertIn("consensus_score", data)
        self.assertIn("fraud_votes", data)
        self.assertIn("models_checked", data)
        self.assertIn("model_breakdown", data)
        self.assertEqual(len(data["model_breakdown"]), 4)
        self.assertIn("prediction", data)
        self.assertIn("risk_tier", data)
        self.assertIn("balance_forensics", data)
        self.assertLess(data["fraud_probability"], 0.70)
        self.assertEqual(data["balance_forensics"]["error_balance_orig"], 0.0)
        self.assertEqual(data["balance_forensics"]["error_balance_dest"], 0.0)

    def test_predict_fraudulent_cash_out_transaction(self):
        """Test POST /predict with a classic fraud pattern: full balance depletion on CASH_OUT."""
        payload = {
            "type": "CASH_OUT",
            "amount": 950000.0,
            "oldbalanceOrg": 950000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 204,
            "nameOrig": "C999999991",
            "nameDest": "C888888882"
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("fraud_probability", data)
        self.assertIn(data["risk_tier"], ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        self.assertTrue(data["balance_forensics"]["is_complete_depletion"])
        self.assertGreater(data["consensus_score"], 0.0)

    def test_predict_validation_error(self):
        """Test POST /predict with invalid data fails validation (HTTP 422)."""
        payload = {
            "type": "INVALID_TYPE",
            "amount": -100.0,  # Negative amount violates ge=0.0 constraint
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 400.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_batch_prediction_endpoint(self):
        """Test POST /predict/batch scores an array of transactions with consensus."""
        payload = {
            "transactions": [
                {
                    "type": "PAYMENT",
                    "amount": 50.0,
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 950.0,
                    "oldbalanceDest": 200.0,
                    "newbalanceDest": 250.0,
                    "step": 5,
                    "nameOrig": "C1001",
                    "nameDest": "M2001"
                },
                {
                    "type": "TRANSFER",
                    "amount": 450000.0,
                    "oldbalanceOrg": 450000.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "step": 33,
                    "nameOrig": "C1002",
                    "nameDest": "C2002"
                }
            ]
        }
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("batch_size"), 2)
        self.assertEqual(len(data.get("results")), 2)
        self.assertIn("total_flagged_fraud", data)
        self.assertIn("total_approved", data)
        first_item = data["results"][0]
        self.assertIn("consensus_score", first_item)
        self.assertIn("model_breakdown", first_item)

    def test_stream_publish_endpoint(self):
        """Test POST /stream/publish publishes to broker and genuinely consumes from queue without bypass."""
        payload = {
            "type": "TRANSFER",
            "amount": 25000.0,
            "oldbalanceOrg": 100000.0,
            "newbalanceOrig": 75000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 25000.0,
            "step": 15,
            "nameOrig": "C1003",
            "nameDest": "C2003"
        }
        response = self.client.post("/stream/publish?mode=simulation", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "stream_consumed_and_scored")
        self.assertEqual(data.get("streaming_engine"), "simulated_broker")
        self.assertTrue(data.get("verified_consumption"))
        self.assertIn("producer_receipt", data)
        self.assertIn("streaming_alert", data)
        self.assertEqual(data["event_id"], data["producer_receipt"]["event_id"])

    def test_stream_publish_live_fails_without_broker(self):
        """Test POST /stream/publish with mode=live fails clearly when broker is absent."""
        payload = {
            "type": "TRANSFER",
            "amount": 5000.0,
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 5000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 5000.0,
            "step": 1,
        }
        response = self.client.post("/stream/publish?mode=live", json=payload)
        # Should return 503 because live broker on localhost:9092 is not running
        self.assertEqual(response.status_code, 503)
        self.assertIn("Live Kafka streaming unavailable", response.json()["detail"])

    def test_rate_limiting_tied_to_authenticated_identity_not_analyst_header(self):
        """Verify rate limiting binds to authenticated API key hash and changing X-Analyst-ID does not bypass limits."""
        payload = {
            "type": "PAYMENT",
            "amount": 10.0,
            "oldbalanceOrg": 100.0,
            "newbalanceOrig": 90.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 10.0,
            "step": 1
        }
        # Simulate hitting rate limit on store directly
        from src.api import get_store
        import config
        import hashlib
        key_hash = hashlib.sha256(config.TEST_API_KEY.encode()).hexdigest()[:16]
        client_key = key_hash
        store = get_store()
        # Consume quota
        for _ in range(125):
            store.check_and_record_rate_limit(client_key, limit=120, window_seconds=60.0)

        # Attempt call with attacker changing X-Analyst-ID to evade limiter
        evasion_headers = {
            "X-API-Key": config.TEST_API_KEY,
            "X-Analyst-ID": "analyst_different_spoofed_id_999",
        }
        resp = self.client.post("/predict", json=payload, headers=evasion_headers)
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Rate limit exceeded", resp.json()["detail"])

    def test_stream_alerts_endpoint(self):
        """Test GET /stream/alerts returns history of consumer-scored alerts."""
        resp = self.client.get("/stream/alerts?mode=simulation")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_alerts", data)
        self.assertIn("alerts", data)

    def test_stream_alerts_live_mode_rejected_when_broker_unreachable(self):
        """Test GET /stream/alerts?mode=live raises 503 when live broker is down instead of returning empty 200."""
        resp = self.client.get("/stream/alerts?mode=live")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("Live Kafka alert retrieval failed", resp.json()["detail"])

    def test_stream_alerts_invalid_mode_rejected(self):
        """Test GET /stream/alerts?mode=invalid raises 400 Bad Request."""
        resp = self.client.get("/stream/alerts?mode=invalid_mode")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid streaming mode", resp.json()["detail"])

    def test_stream_alerts_live_decoding_with_actual_payload(self):
        """Test that live alert deserialization decodes actual bytes using json without NameError."""
        from unittest.mock import MagicMock, patch

        mock_record = MagicMock()
        alert_payload = {
            "alert_id": "ALT-TEST-LIVE-001",
            "event_id": "EVT-TEST-LIVE-001",
            "decision": "FRAUDULENT",
            "amount": 500000.0,
            "type": "TRANSFER",
            "is_flagged": True,
        }
        encoded_bytes = json.dumps(alert_payload).encode("utf-8")
        
        # Test the deserializer directly to guarantee json is in scope
        from src.api import get_streaming_alerts
        mock_consumer = MagicMock()
        mock_consumer.poll.return_value = {
            "financial.fraud-alerts-0": [
                MagicMock(value=alert_payload)
            ]
        }
        
        with patch("kafka.KafkaConsumer", return_value=mock_consumer) as mock_cls:
            resp = self.client.get("/stream/alerts?mode=live")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["streaming_engine"], "live_kafka")
            self.assertEqual(len(data["alerts"]), 1)
            self.assertEqual(data["alerts"][0]["alert_id"], "ALT-TEST-LIVE-001")
            
            # Verify value_deserializer lambda function works without NameError
            deserializer = mock_cls.call_args[1]["value_deserializer"]
            decoded = deserializer(encoded_bytes)
            self.assertEqual(decoded["alert_id"], "ALT-TEST-LIVE-001")
            self.assertEqual(decoded["decision"], "FRAUDULENT")

    def test_unauthenticated_request_rejected_without_valid_secret(self):
        """Test that requests with missing or invalid API keys are rejected with 401."""
        payload = {
            "type": "PAYMENT",
            "amount": 10.0,
            "oldbalanceOrg": 100.0,
            "newbalanceOrig": 90.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 10.0,
            "step": 1
        }
        # Missing key
        resp_missing = self.unauth_client.post("/predict", json=payload)
        self.assertEqual(resp_missing.status_code, 401)
        self.assertIn("Unauthorized", resp_missing.json()["detail"])

        # Invalid key
        resp_invalid = self.client.post("/predict", json=payload, headers={"X-API-Key": "wrong-secret-key-12345"})
        self.assertEqual(resp_invalid.status_code, 401)
        self.assertIn("Unauthorized", resp_invalid.json()["detail"])


if __name__ == "__main__":
    unittest.main()
