# FraudGuard AI — Financial Fraud Detection with ML and DL

A local academic demonstration comparing two machine-learning models and two
deep-learning models, with a Streamlit dashboard for single and batch scoring.

## Sign in and keep your work

Open the dashboard and choose **Create account**. Enter your name, email login
identifier and a password/passphrase with at least 12 characters. New accounts
sign in immediately; existing users can use the **Sign in** tab.

In **My Workspace**, upload and name a transaction CSV, or choose **Add demo
dataset**. Open a saved dataset to use it in Data Analysis. In **Fraud Prediction
& Alerts**, choose Batch CSV and select the saved workspace dataset to score it
with any of the four trained models. Single-transaction predictions and batch
results save automatically in **My Workspace → Analysis history**.

## Compare every model and inspect a flagged transaction

**Fraud Prediction & Alerts** now defaults to **Compare all models**. Random
Forest, XGBoost, Deep Neural Network and Deep Autoencoder each score the same
input using their own saved validation threshold. Select an individual model
from the same menu to use it alone. Scores from different models are not averaged.

For a single transaction, the comparison shows each model's verdict, score and
threshold. For a batch, each source row keeps its identifier and gains four sets
of model results, `Fraud_Votes`, `Models_Checked`, `Flagged_By`, and an explicit
review status. A row is **FLAGGED FOR REVIEW** when at least one model flags it;
this is a transparent review rule, not a separately trained ensemble classifier.

The flagged-only filter highlights rows needing review. Choose **Transaction to
inspect** to see that row's model decisions and quantitative balance checks:
sender debit, amount-versus-balance ratio, recipient credit, and the engineered
balance discrepancies. Clearing the filter lets you inspect unflagged rows too.
Any supplied `isFraud` label appears separately; it never determines a prediction.
Rows labelled fraud but missed by the models are counted explicitly.

Example from the supplied synthetic CSV: **row 5**, a **113,292.92-unit TRANSFER**,
is flagged by all four models. The sender opening balance is 116,909.29 and the
closing balance is 13,070.90, so the recorded debit is 103,838.39. That differs
from the transaction amount by 9,454.53. The dataset supplies `isFraud=1` for this
example. These balance observations are review context, not causal model
attributions or standalone proof of fraud; each model's threshold comparison is
shown as the precise reason for its flag.

Signing out clears the browser session. Signing back in restores the last-opened
dataset and gives access to previous datasets, scored results and CSV downloads.
Reloading the browser may require signing in again; saved work remains on disk.
Workspace uploads are scored using the existing models and do not retrain them.

Accounts and work are stored in `storage/workspaces.sqlite3` on this computer.
Each account can retrieve only its own saved datasets and analyses. Passwords
use random salts and PBKDF2-HMAC-SHA256 with 600,000 iterations. Uploaded data and
results are stored unencrypted in the local database. Accounts have no email
verification or password-recovery service. This is local sign-in, not a cloud
account system; work does not sync to another computer.

`storage/` is excluded from Git and the submission archive, so personal accounts
and uploaded data are never bundled into the submission ZIP. Keep this folder
when restarting or updating the project to retain saved work.

## Implemented models

| Model | Family | Training objective |
|---|---|---|
| Random Forest | ML | Supervised tree ensemble |
| XGBoost | ML | Supervised gradient-boosted trees |
| Deep Neural Network | DL | Three hidden layers (64, 32, 16), ReLU, sigmoid output, binary cross-entropy, Adam |
| Deep Autoencoder | DL | Five hidden layers (64, 32, 8, 32, 64), ReLU, linear reconstruction output, MSE, Adam |

The deep models use scikit-learn's `MLPClassifier` and `MLPRegressor`
backpropagation implementations. They are trained networks with multiple hidden
layers, saved weights, learning histories, and restored validation checkpoints.
TensorFlow, PyTorch, a GPU, and sequence data are not required.

## Verified results

All **104 automated checks passed**, with zero failures and zero skipped tests. These include sign-in,
all seven dashboard pages, per-account data isolation, saved dataset/history
reopening after a fresh sign-in, scoring with all four models, saved-model
reloads, input validation, preserved duplicate inference rows, training-only preprocessing,
four-model score/threshold comparisons, disagreement cases, transaction explanations,
real-time Kafka streaming queue consumption and DLQ routing, consumer manual offset commits and crash recovery/idempotency,
live alerts consumption with broker availability checks,
persistent multi-process SQLite rate limiting bound to authenticated identities, and exact saved comparison CSV round trips.
The full report is `results/test_report.txt` and summary in `results/test_summary.json`.

Results on 3,000 held-out synthetic transactions (45 fraud examples):

| Model | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| Random Forest | 0.8235 | 0.9333 | 0.8750 | 0.9581 |
| XGBoost | 0.8889 | 0.8889 | 0.8889 | 0.9418 |
| Deep Neural Network | 1.0000 | 0.8889 | 0.9412 | 0.9307 |
| Deep Autoencoder | 0.9762 | 0.9111 | 0.9425 | 0.9586 |

XGBoost is the validation-selected benchmark model because it ranked first on
**validation** average precision. The prediction page defaults to comparing all
four models. The final test comparison is reported separately and does not
override the benchmark selection.

## Start the project

On this computer, double-click **START_PROJECT.bat**, or run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open http://localhost:8501. The trained artifacts are already stored in `models/`.

For a fresh installation, use Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

`requirements.txt` lists the direct dependencies; `requirements-lock.txt` records
the versions used for the verified run. Training regenerates models, charts,
sample transactions, and experiment metadata.

To regenerate the test report, run `.\.venv\Scripts\python.exe scripts/run_tests.py`.

## Real-Time Kafka Streaming Pipeline

FraudGuard AI provides an end-to-end streaming ingestion, broker queue consumption, and fraud scoring pipeline (`src/kafka_streaming.py`) supporting both live Apache Kafka clusters and a zero-dependency in-memory `SimulatedStreamingBroker` for standalone demonstrations:

- **TransactionStreamProducer:** Validates raw transaction schemas, formats JSON event envelopes with timestamps and deterministic keys, and streams to `financial.transactions`.
- **FraudStreamingConsumer:** Actively consumes and drains events from the topic queue in real-time, scores them using the four-model consensus engine (or individual models), computes forensic balance factors, tracks inference latency (ms), and publishes structured alerts to `financial.fraud-alerts`.
- **Crash Recovery & Idempotency:** If a consumer crashes before offset commit, re-processing the same event is detected via the SQLite database case store (`is_duplicate=True`), preventing duplicate case creation or ledger inconsistency.
- **Dead-Letter Queue (DLQ):** Poisonous or malformed transaction events are routed to `financial.transactions.dlq` with failure diagnostics without halting consumer operation.
- **Run the Streaming Demo:**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_kafka_streaming.py --events 30 --rate 10 --mode simulated
  ```
  CLI arguments include `--mode {simulated, live}`, `--events <int>`, `--rate <float>`, `--model {comparison, rf, xgb, dnn, ae}`, and `--broker <host:port>`.

## Experimental method

1. Read `data/transactions.csv` and validate raw transaction fields and labels.
2. Remove duplicate labelled examples before partitioning. Ignore rule flags,
   labels, transaction IDs, and other metadata as model inputs.
3. Make stratified **60% training / 20% validation / 20% final test** partitions
   using random seed 42. Save source row indices to `results/split_indices.json`.
4. Fit numeric median imputation, transaction-type mode imputation, and
   `RobustScaler` on training rows only. Reuse those parameters during inference.
   The five supported transaction categories form a fixed one-hot schema.
5. Apply **SMOTE-NC** to training data only. The transaction type is treated as
   one categorical variable so synthetic samples keep a valid category.
6. Train Random Forest, XGBoost, and the deep classifier on balanced training
   data. Train the autoencoder on **original legitimate training rows only**.
7. The deep models apply an additional training-fitted `StandardScaler` and clip
   standardized inputs to [-12, 12] for the classifier and [-20, 20] for the
   autoencoder. Classifier early stopping monitors validation average precision;
   autoencoder early stopping monitors legitimate validation reconstruction MSE.
8. Select each model's decision threshold by maximum validation F1. Select the
   default model by validation average precision, breaking ties by F1 and recall.
   The autoencoder representation is learned without fraud examples; its final
   operating threshold uses labelled validation data.
9. Evaluate all four models on the untouched final test set. Report precision,
   recall, F1, ROC-AUC, average precision (the PR-AUC column), specificity and
   confusion counts. No test metrics determine model or threshold selection.
10. Generate tree SHAP explanations for the best validation-ranked tree model.
    These plots explain that tree model, even if the overall winner is neural.

## Dataset and limitations

The existing **15,000 transactions with 225 fraud labels** are a locally generated
synthetic benchmark. They are not real bank transactions and are not an official
PaySim dataset. These results establish performance only on this generator's
distribution. No claim of independent real-world validation is made.

Required input fields:

```text
step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
```

Supported types: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`.
Missing numeric values use training medians; missing types use the training mode.
Invalid strings, infinity, negative monetary values, unknown categories, missing
columns and empty batches produce descriptive errors. Inference preserves all
input rows and indices, including repeated transactions.

The dashboard displays **scores**, not calibrated probabilities. The autoencoder
score is `error / (error + training_normal_error_99th_percentile)`. Its
`predict_proba` method and the legacy `Fraud_Probability` CSV column exist for
interface compatibility; they do not imply probability calibration.

The 30%/60%/80% LOW/MEDIUM/HIGH/CRITICAL score bands are illustrative and separate
from each model's validation-selected binary decision threshold. All suggested
actions are advisory; no transaction is blocked or cleared by this software.

## Operational Framing: Retrospective Review vs. Pre-Authorization Blocking

A critical architectural distinction in this system is its operational timing:
* **Post-Transaction Audit & Investigation Queue:** The model input schema requires closing balances (`newbalanceOrig`, `newbalanceDest`). In real-world core banking and card networks, closing balances describe settled or completed transactions. Consequently, FraudGuard AI is engineered for retrospective risk triage, compliance screening, and forensic analyst adjudication queues immediately following ledger posting or streaming event ingestion.
* **Pre-Authorization Fraud Blocking Limitation:** Detecting fraud *prior* to transaction authorization requires pre-authorization features (e.g., velocity, geolocation, device fingerprinting, and opening balance alone). Because closing balances cannot be observed before payment completion, this system should not be characterized as a pre-authorization payment blocking gateway.

## Dataset Transparency & Real-World Accuracy Limitations

1. **Local Synthetic Training Partition (15,000 records):**
   * The primary models were trained on 15,000 synthetic transactions (`data/transactions.csv`) generated by rule-based simulation.
   * High performance on this dataset reflects the models learning the underlying generator's mathematical balance debit and credit rules.
   * These synthetic metrics are explicitly documented as an academic baseline; **they do not establish real-world accuracy on actual bank transactions.**

2. **Secondary Synthetic Distribution Shift Benchmark (PaySim-schema compatible):**
   * **Dataset Provenance & Sampling Procedure:**
     * Generated locally using NumPy pseudorandom distributions (`seed=42`) via `scripts/generate_small_dataset.py`.
     * `data/paysim_benchmark_12mb.csv` contains 120,000 synthetic rows (~12 MB).
     * `data/paysim_sample_1000.csv` is the deterministically sampled first 1,000 transactions from this generation, containing exactly 15 fraud instances (1.5% fraud prevalence).
     * **Provenance Notice:** Adhering to the 11 columns and transaction types of the PaySim schema does **not** make this an official PaySim release from Kaggle or external research literature. It is a secondary synthetic dataset designed to evaluate model degradation under domain shift without requiring a 500 MB download.
   * **Operational Performance at Frozen Decision Thresholds:**
     The models (trained exclusively on primary data) were evaluated at their saved validation decision thresholds against the 1,000-row secondary synthetic sample (`results/independent_benchmark_report.json`):

     | Model | Family | Saved Threshold | Fraud Caught (out of 15) | False Alarms | Precision | Recall | ROC-AUC |
     |---|---|---:|---:|---:|---:|---:|---:|
     | **Random Forest** | Machine Learning | 0.740 | **0 / 15** | 6 | 0.00% | 0.00% | 0.8783 |
     | **XGBoost** | Machine Learning | 0.945 | **0 / 15** | 146 | 0.00% | 0.00% | 0.8251 |
     | **Deep Neural Network** | Deep Learning | 0.999 | **15 / 15** | 87 | 14.71% | 100.00% | 0.9635 |
     | **Deep Autoencoder** | Deep Learning | 0.871 | **15 / 15** | 336 | 4.27% | 100.00% | 0.9118 |

   * **Key Operational Limitations & Analysis:**
     * **No Claim of Robust Transferability:** Reporting only ROC-AUC (0.82–0.96) conceals catastrophic operational triage failure at the frozen thresholds.
     * **Tree Model Collapse:** Random Forest and XGBoost caught 0/15 fraud cases (0% recall). Because tree splits tightly fit the exact balance debit/credit rules of the primary generator, subtle differences in secondary synthetic distributions cause complete recall collapse.
     * **Neural Network False Alarms:** The Deep Neural Network and Autoencoder maintained 100% sensitivity (15/15 fraud detected) but incurred massive false alarm rates (87 and 336 false alarms, yielding 14.7% and 4.3% precision respectively). In a production bank triage workflow, a 4–14% precision rate would overwhelm forensic investigators.
     * **Methodological Conclusion:** High ROC-AUC reflects ranking ability on score distributions, but operational decisions require calibrated thresholds. Deploying models to new environments requires re-calibrating decision thresholds on an in-domain validation split before production use.
   * Run the evaluation script anytime via:
     ```powershell
     python scripts/evaluate_independent_benchmark.py data/paysim_sample_1000.csv
     ```

## API Gateway & Security Architecture

The FastAPI microservice (`src/api.py`) exposes production-grade REST endpoints with defense-in-depth:
* **Multi-Model Consensus Scoring:** `/predict` and `/predict/batch` evaluate all 4 models simultaneously, returning true consensus fraction (`consensus_score`), vote count (`fraud_votes`), and individual model breakdowns.
* **API Key Authentication & Secret Isolation:**
  * Requires environment-provided secrets (`FRAUDGUARD_API_KEY` or `FRAUDGUARD_API_KEYS`).
  * No publicly known default API keys are hardcoded in the codebase for production deployment.
  * In non-production testing, an isolated offline test key (`test-analyst-secret-offline`) is accepted strictly for automated CI execution.
* **Persistent Multi-Process Rate Limiting:**
  * Rate limits (120 req/min) are strictly bound to the authenticated identity (`sha256(raw_key)[:16]`). An attacker altering the caller-supplied `X-Analyst-ID` header **cannot** reset or bypass their rate limit counter.
  * Counter state is persisted in SQLite (`storage/workspaces.sqlite3`) using WAL mode and transaction locking, ensuring rate limits are strictly shared and synchronized across multi-worker Uvicorn processes.
* **Live Kafka Streaming & Broker Consumption:**
  * `/stream/publish` publishes events to `financial.transactions` and consumes them directly from the broker topic queue using `FraudStreamingConsumer.consume_one()`.
  * In live mode (`mode=live`), connects to external Apache Kafka (`localhost:9092`), failing with clear HTTP 503 if the cluster is unreachable.
  * `/stream/alerts` provides programmatic access to recent alerts published by the consumer.

## Public Repository Alignment & Scope Notice

* **Active Implemented Models (4):** Random Forest, XGBoost, Deep Neural Network (MLP), and Deep Autoencoder (Reconstruction Anomaly Detector).
* **Explainability:** Tree SHAP is implemented for gradient-boosted decision trees (`src/explainability.py`).
* **Notice on LSTM & LIME:** Earlier exploratory code references to sequential LSTM modeling and perturbation-based LIME are documented architectural baselines / future extensions, rather than components of the active 4-model production comparison pipeline.
* **Recommended GitHub Repository Description:**
  > *Multi-model retrospective fraud detection system (XGBoost, Random Forest, Deep Neural Network, Deep Autoencoder) with Streamlit analyst workspace, FastAPI gateway, Kafka streaming, and Tree SHAP explainability.*

## Outputs

| Location | Contents |
|---|---|
| `models/` | Four trained models, shared preprocessor and model/threshold manifest |
| `results/model_comparison.csv` | Final held-out test metrics for all models |
| `results/validation_comparison.csv` | Validation metrics and operating thresholds |
| `results/test_predictions.csv` | Per-row held-out scores and predicted labels |
| `results/experiment_metadata.json` | Dataset hash, splits, versions, architectures, timings and selection rules |
| `results/deep_learning_training.png` | Neural validation learning curves and checkpoint epochs |
| `results/*_history.csv` | Per-epoch neural training records |
| `results/*.png` | Confusion matrices, ROC/PR curves, model comparison, SMOTE-NC and SHAP plots |
| `data/sample_test.csv` | 30 held-out demo examples, including both fraud and legitimate labels |
| `logs/pipeline.log` | Pipeline log |

## Demonstration sequence

1. Create an account or sign in. In My Workspace, add the demo dataset or upload a CSV.
2. Open Data Analysis to inspect the selected dataset, then Overview to identify the two ML and two DL models.
3. Open Model Performance and compare benchmark fraud recall, precision and average precision.
4. Open Deep Learning to show both architectures and actual validation curves.
5. Open Fraud Prediction & Alerts, keep Compare all models selected, then scan the saved dataset in Batch CSV mode. Select a flagged transaction to inspect its model decisions and balance checks.
6. Sign out, sign in again, and reopen the same dataset and saved analysis from My Workspace.
7. Open About & Architecture to explain the offline training and saved inference workspace diagrams.

The original source and artifacts were preserved in
`project_backup_before_ml_dl.zip` before implementation changes.


## Deployment secrets and streaming recovery

Docker Compose requires `FRAUDGUARD_API_KEY` and refuses to start without it.
Generate a private value with `python -c "import secrets; print(secrets.token_urlsafe(32))"`,
set it in your shell, and send it in the `X-API-Key` header. Never commit the value.
For example, in PowerShell set `$env:FRAUDGUARD_API_KEY = "YOUR_PRIVATE_VALUE"`
before running `docker compose up --build`.

Kafka uses `kafka:29092` inside Docker and `localhost:9092` from the host.
The API waits for publication acknowledgement before reporting an event as
published. A timeout or a different consumed event produces
`published_pending_consumption`, not a claim that the requested event was scored.
Keep a consumer running to drain pending input; the synchronous endpoint only
polls briefly and does not guarantee immediate processing.

Flagged cases and their pending alerts are saved in one SQLite transaction.
On replay, the consumer retries any pending alert before committing the input
offset. Alert and dead-letter publication wait for broker acknowledgement;
processing or commit failures rewind the consumer position for retry.
Alerts carry stable IDs derived from the input event ID. Delivery is **at least
once**, not exactly once: a crash after broker acknowledgement but before the
SQLite delivery marker can cause redelivery. Alert consumers must deduplicate
by `alert_id` or `event_id`. Existing cases created before this change are not
retroactively assigned new alerts.

`tests/test_delivery_reliability.py` covers transactional rollback, restart
recovery, failed acknowledgements, commit failures, event correlation, live
alert JSON decoding, and the required Compose secret. Kafka clients are doubled
in these automated tests. They are not proof of a real Kafka deployment:
Docker and a live broker were unavailable in the review environment. Run a
broker-backed publish, consume, restart and replay exercise before deployment.
The changes do not alter the models or establish real-world fraud accuracy.
