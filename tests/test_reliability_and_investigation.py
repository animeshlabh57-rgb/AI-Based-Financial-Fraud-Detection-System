"""
Rigorous Reliability, Security, and Investigation Workflow Test Suite.

Proves:
1. Crash-and-Replay Idempotency: Consumer crash after DB write but before offset commit;
   replaying the event creates ZERO duplicate cases or duplicate audit records.
2. Investigation Case Queue: Full case lifecycle (OPEN -> IN_REVIEW -> RESOLVED: CONFIRMED_FRAUD)
   with immutable chronological audit trail.
3. Cross-Tenant Isolation: Users cannot read or modify another account's datasets or runs.
4. Concurrent Database Operations: High-concurrency reads and writes using WAL mode.
5. Dead-Letter Queue (DLQ): Poison messages routed to DLQ without service disruption.
6. Model Versioning & Rollback: Atomic operational model rollback with provenance tagging.
"""

import concurrent.futures
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config
from workspace_store import WorkspaceStore
from kafka_streaming import (
    FraudStreamingConsumer,
    SimulatedStreamingBroker,
    TransactionStreamProducer,
)
from model_registry import ModelRegistry
from cost_simulator import CostThresholdSimulator


class TestReliabilityAndInvestigation(unittest.TestCase):
    """Test suite covering enterprise reliability, case management, and recovery evidence."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="fraudguard-test-")
        self.db_path = os.path.join(self.temp_dir.name, "test_workspace.sqlite3")
        self.store = WorkspaceStore(self.db_path)
        self.broker = SimulatedStreamingBroker()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_case_lifecycle_and_audit_trail(self):
        """Verify full case lifecycle and immutable audit trail."""
        event_id = "EVT-TEST-0001"
        case, created = self.store.create_or_get_case(
            event_id=event_id,
            transaction_id="TXN-9988",
            amount=450000.0,
            txn_type="CASH_OUT",
            risk_score=0.985,
            risk_tier="CRITICAL",
            decision="FLAGGED FOR REVIEW",
            assigned_analyst="Unassigned",
            explanation={"top_risk_factors": ["Total balance depletion"]},
        )
        self.assertTrue(created)
        case_id = case["case_id"]

        # Step 1: Assign to analyst & set IN_REVIEW
        updated1 = self.store.update_case(
            case_id=case_id,
            changed_by="Senior Analyst Sarah",
            assigned_analyst="Sarah Jenkins",
            status="IN_REVIEW",
            notes="Under manual review. Cross-referencing mule destination account.",
        )
        self.assertEqual(updated1["status"], "IN_REVIEW")
        self.assertEqual(updated1["assigned_analyst"], "Sarah Jenkins")

        # Step 2: Final adjudication (RESOLVED: CONFIRMED_FRAUD)
        updated2 = self.store.update_case(
            case_id=case_id,
            changed_by="Senior Analyst Sarah",
            status="RESOLVED",
            outcome="CONFIRMED_FRAUD",
            notes="Confirmed account takeover fraud. Destination account frozen.",
        )
        self.assertEqual(updated2["status"], "RESOLVED")
        self.assertEqual(updated2["outcome"], "CONFIRMED_FRAUD")

        # Step 3: Verify chronological audit trail
        audit_log = self.store.get_case_audit_log(case_id)
        self.assertEqual(len(audit_log), 3)  # CREATED, UPDATED (IN_REVIEW), UPDATED (RESOLVED)
        self.assertEqual(audit_log[0]["action"], "CASE_CREATED")
        self.assertEqual(audit_log[1]["action"], "CASE_UPDATED")
        self.assertEqual(audit_log[2]["action"], "CASE_UPDATED")
        self.assertIn("Sarah Jenkins", audit_log[1]["new_state"])
        self.assertIn("CONFIRMED_FRAUD", audit_log[2]["new_state"])

    def test_crash_and_replay_idempotency(self):
        """
        Simulate a worker crash after writing a prediction to the database
        but BEFORE committing its Kafka offset. Replaying the event must produce
        ZERO duplicate cases or duplicate audit records.
        """
        producer = TransactionStreamProducer(use_simulated=True, simulated_broker=self.broker)
        crashing_consumer = FraudStreamingConsumer(
            use_simulated=True,
            simulated_broker=self.broker,
            store=self.store,
            simulate_crash_before_commit=True,  # Simulate crash hook
        )

        sample_tx = {
            "Transaction_ID": "TXN-CRASH-01",
            "type": "CASH_OUT",
            "amount": 900000.0,
            "oldbalanceOrg": 900000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 45,
        }

        # 1. Produce message
        receipt = producer.produce(sample_tx, key="TXN-CRASH-01")
        event_id = receipt["event_id"]

        # 2. Worker processes message, writes to DB, but CRASHES before committing offset
        with self.assertRaises(RuntimeError) as ctx:
            crashing_consumer.process_message(receipt)
        self.assertIn("SimulatedConsumerCrash", str(ctx.exception))

        # Check DB state after crash: 1 case was saved
        initial_case = self.store.get_case_by_event(event_id)
        self.assertIsNotNone(initial_case)
        case_id = initial_case["case_id"]
        audit_records_before = self.store.get_case_audit_log(case_id)
        self.assertEqual(len(audit_records_before), 1)

        # Check that pending alert was persisted alongside the case before crash
        pending_alert_before = self.store.get_pending_alert_by_event(event_id)
        self.assertIsNotNone(pending_alert_before)
        self.assertEqual(pending_alert_before["status"], "PENDING")
        self.assertEqual(self.broker.topic_size(config.KAFKA_ALERTS_TOPIC), 0, "No alert should have been sent before crash!")

        # 3. Consumer restarts (simulated recovery) and re-processes uncommitted event
        recovering_consumer = FraudStreamingConsumer(
            alerts_topic=config.KAFKA_ALERTS_TOPIC,
            use_simulated=True,
            simulated_broker=self.broker,
            store=self.store,
            simulate_crash_before_commit=False,
        )

        replay_result = recovering_consumer.process_message(receipt)

        # 4. Verify recovery delivers the pending alert and maintains idempotency
        self.assertTrue(replay_result["is_duplicate"])
        self.assertEqual(replay_result["idempotent_action"], "DELIVERED_PENDING_ALERT_ON_REPLAY")
        self.assertTrue(replay_result["alert_delivered_on_recovery"])
        self.assertEqual(replay_result["case_id"], case_id)

        # Confirm the pending alert is now marked DELIVERED and alert sits in topic
        pending_alert_after = self.store.get_pending_alert_by_event(event_id)
        self.assertEqual(pending_alert_after["status"], "DELIVERED")
        self.assertEqual(self.broker.topic_size(config.KAFKA_ALERTS_TOPIC), 1, "Recovery must deliver the alert!")

        # Replaying again after delivery skips completely without duplicate alert
        replayed_again = recovering_consumer.process_message(receipt)
        self.assertEqual(replayed_again["idempotent_action"], "SKIPPED_EXISTING_CASE")
        self.assertEqual(self.broker.topic_size(config.KAFKA_ALERTS_TOPIC), 1, "Must not publish duplicate alert on second replay!")

        # Confirm ZERO duplicate records in database
        all_cases = self.store.list_cases(limit=100)
        self.assertEqual(len(all_cases), 1, "There must be exactly 1 case, not 2!")

        audit_records_after = self.store.get_case_audit_log(case_id)
        self.assertEqual(len(audit_records_after), 1, "Audit trail must not record duplicate creation entries!")

    def test_dead_letter_queue_on_poison_message(self):
        """Verify invalid/poisonous message is routed to DLQ without crashing consumer."""
        consumer = FraudStreamingConsumer(
            use_simulated=True,
            simulated_broker=self.broker,
            store=self.store,
        )

        poison_message = {
            "event_id": "EVT-POISON-666",
            "transaction": {
                "amount": "NOT_A_NUMBER",  # Poison payload
                "type": "INVALID_TYPE",
            },
        }

        result = consumer.process_message(poison_message)
        self.assertIn("dlq_id", result)
        self.assertEqual(consumer.dlq_published, 1)

        # Verify DLQ topic contains the failed message
        dlq_history = self.broker.get_history(config.KAFKA_DLQ_TOPIC)
        self.assertGreaterEqual(len(dlq_history), 1)
        dlq_deserialized = dlq_history[-1].deserialize()
        self.assertEqual(dlq_deserialized["failed_message"]["event_id"], "EVT-POISON-666")

    def test_cross_tenant_isolation(self):
        """Verify Tenant A cannot access Tenant B's data or analysis runs."""
        user_a = self.store.create_user("Analyst Alpha", "alpha@bank.com", "SecurePassAlpha123!")
        user_b = self.store.create_user("Analyst Beta", "beta@bank.com", "SecurePassBeta123!!")

        csv_content = b"step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n1,PAYMENT,10.0,100.0,90.0,0.0,10.0"
        dataset_a = self.store.save_dataset(user_a["id"], "Alpha Secret Dataset", csv_content)

        run_a = self.store.save_run(
            user_id=user_a["id"],
            dataset_id=dataset_a["id"],
            model_name="XGBoost",
            result_csv=b"Transaction_ID,Score\nTXN-001,0.95\n",
            summary={"score": 0.95},
            kind="single",
        )

        # Tenant B attempts to read Tenant A's dataset and run
        self.assertIsNone(self.store.get_dataset(user_b["id"], dataset_a["id"]))
        self.assertIsNone(self.store.get_run(user_b["id"], run_a["id"]))

        # Tenant B attempts to submit a review on Tenant A's run
        with self.assertRaises(ValueError):
            self.store.save_review(
                user_id=user_b["id"],
                run_id=run_a["id"],
                source_row=1,
                status="Reviewed",
                notes="Unauthorized modification",
                reviewer_display_name="Analyst Beta",
            )

    def test_concurrent_writes_wal_mode(self):
        """Test high-concurrency case creation across multiple threads under WAL mode."""
        def _worker(thread_idx):
            with WorkspaceStore(self.db_path) as local_store:
                for i in range(15):
                    evt_id = f"EVT-CONCUR-{thread_idx}-{i}"
                    local_store.create_or_get_case(
                        event_id=evt_id,
                        transaction_id=f"TXN-{thread_idx}-{i}",
                        amount=100.0 * (i + 1),
                        txn_type="PAYMENT",
                        risk_score=0.1,
                        risk_tier="LOW",
                        decision="LEGITIMATE",
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_worker, t) for t in range(4)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Will raise if any thread encountered locking errors

        total_cases = len(self.store.list_cases(limit=1000))
        self.assertEqual(total_cases, 60)  # 4 threads * 15 cases = 60

    def test_model_registry_and_rollback(self):
        """Verify model version registration and atomic rollback."""
        reg_path = os.path.join(self.temp_dir.name, "model_registry.json")
        registry = ModelRegistry(registry_path=reg_path)

        active = registry.get_active_version()
        self.assertEqual(active["version_id"], "v1.0.0-xgb-consensus")

        # Rollback to v0.9.0-rf-baseline
        rollback_receipt = registry.rollback_to_version("v0.9.0-rf-baseline", operator_name="Lead Analyst Dhruv")
        self.assertEqual(rollback_receipt["status"], "ROLLBACK_SUCCESSFUL")
        self.assertEqual(rollback_receipt["active_version"], "v0.9.0-rf-baseline")

        # Verify active version changed
        new_active = registry.get_active_version()
        self.assertEqual(new_active["version_id"], "v0.9.0-rf-baseline")
        self.assertEqual(new_active["model_name"], "Random Forest")

        # Provenance stamping test
        dummy_pred = {"Transaction_ID": "TXN-REG-01", "Fraud_Probability": 0.85}
        stamped = registry.stamp_prediction_provenance(dummy_pred)
        self.assertEqual(stamped["model_version"], "v0.9.0-rf-baseline")
        self.assertEqual(stamped["active_threshold"], 0.74)

    def test_cost_threshold_simulator(self):
        """Verify threshold cost simulation and optimal threshold computation."""
        simulator = CostThresholdSimulator(
            cost_missed_fraud=1000.0,
            cost_analyst_review=25.0,
            cost_customer_friction=10.0,
        )

        y_true = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.85, 0.90, 0.35, 0.40, 0.95, 0.05])

        eval_at_half = simulator.evaluate_threshold(y_true, y_prob, threshold=0.50)
        self.assertEqual(eval_at_half["detected_fraud_tp"], 3)
        self.assertEqual(eval_at_half["missed_fraud_fn"], 0)
        self.assertEqual(eval_at_half["false_alarms_fp"], 0)
        self.assertEqual(eval_at_half["financial_breakdown"]["direct_fraud_loss"], 0.0)

        cost_df, best_eval = simulator.simulate_cost_curve(y_true, y_prob, steps=20)
        self.assertGreater(len(cost_df), 0)
        self.assertIsNotNone(best_eval)
        self.assertGreaterEqual(best_eval["threshold"], 0.01)


if __name__ == "__main__":
    unittest.main()
