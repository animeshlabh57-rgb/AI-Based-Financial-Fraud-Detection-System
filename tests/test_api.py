"""Integration and unit tests for FraudGuard AI FastAPI RESTful microservice.
Tests cover health check, single transaction prediction, batch prediction,
Kafka streaming simulation endpoint, and model metadata inspection.
"""

import unittest
from fastapi.testclient import TestClient
from src.api import app


class TestFraudDetectionAPI(unittest.TestCase):
    """Test suite for the FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize the FastAPI test client once for the suite."""
        cls.client = TestClient(app)

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
        """Test POST /predict/batch scores an array of transactions."""
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

    def test_stream_publish_endpoint(self):
        """Test POST /stream/publish simulates publishing transaction to Kafka stream."""
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
        response = self.client.post("/stream/publish", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "stream_processed")
        self.assertIn("producer_receipt", data)
        self.assertIn("streaming_alert", data)


if __name__ == "__main__":
    unittest.main()
