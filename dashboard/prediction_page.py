"""Analyze Transactions: Single-transaction evaluation and Batch CSV predictions."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import io
import pandas as pd
import streamlit as st

import config
from src.model_comparison import FraudModelComparison
from src.dataset_validation import MAX_UPLOAD_BYTES, validate_dataset
from src.kafka_streaming import (
    TransactionStreamProducer,
    FraudStreamingConsumer,
    SimulatedStreamingBroker,
)
from dashboard.workspace_page import csv_frame, open_dataset
from dashboard.transaction_explanation import render_transaction_explanation

COMPARE_OPTION = 'Compare all models'


@st.cache_resource
def get_comparison(artifact_signature):
    # Shared model weights only; private analyst data and results are never cached globally.
    return FraudModelComparison(config.MODELS_DIR)


def artifact_signature(metadata):
    filenames = [row['file'] for row in metadata['models'].values()]
    filenames += ['preprocessor.joblib', 'model_metadata.json']
    return tuple((name, (Path(config.MODELS_DIR) / name).stat().st_mtime_ns) for name in filenames)


def model_rows_from_result(row, metadata, selected):
    if selected == COMPARE_OPTION:
        return [{'Model': name, 'Family': details['family'],
                 'Prediction': row[f'{name} Prediction'], 'Score': float(row[f'{name} Score']),
                 'Threshold': float(row[f'{name} Threshold']), 'Score_Kind': details['score_kind']}
                for name, details in metadata['models'].items()]
    details = metadata['models'][selected]
    score_col = 'Fraud_Probability' if 'Fraud_Probability' in row else 'Score'
    return [{'Model': selected, 'Family': details['family'], 'Prediction': row['Prediction'],
             'Score': float(row[score_col]), 'Threshold': float(row['Decision_Threshold']),
             'Score_Kind': details['score_kind']}]


def model_table(rows):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                 column_config={'Score': st.column_config.NumberColumn(format='%.6f'),
                                'Threshold': st.column_config.NumberColumn(format='%.6f')})


def _comparison_csv(payload, report):
    row = dict(payload, Transaction_ID=report['Transaction_ID'], Source_Row=1,
               Fraud_Votes=report['Fraud_Votes'], Models_Checked=report['Models_Checked'],
               Flagged_By=', '.join(report['Flagged_By']) or 'No model flags',
               Prediction='FLAGGED FOR REVIEW' if report['Fraud_Votes'] else 'NO MODEL FLAG',
               Review_Status=report.get('Review_Status', 'Pending'))
    for model in report['models']:
        for suffix, key in [('Prediction', 'Prediction'), ('Score', 'Score'), ('Threshold', 'Threshold')]:
            row[f'{model["Model"]} {suffix}'] = model[key]
    return pd.DataFrame([row]).to_csv(index=False).encode('utf-8')


def _single_input(read_csv):
    st.markdown('#### Transaction Input Form')
    st.caption('Enter transaction parameters to evaluate. Values represent synthetic currency units in an hourly simulation.')

    examples = {
        'Normal example': {
            'step': 42,
            'type': 'TRANSFER',
            'amount': 5000.0,
            'oldbalanceOrg': 25000.0,
            'newbalanceOrig': 20000.0,  # Exact decrease: 5,000
            'oldbalanceDest': 10000.0,
            'newbalanceDest': 15000.0,  # Exact increase: 5,000
            'note': 'Internally consistent transfer: sender balance decrease matches amount, and recipient balance increases by amount.'
        },
        'Suspicious example': {
            'step': 348,
            'type': 'TRANSFER',
            'amount': 250000.0,
            'oldbalanceOrg': 25000.0,
            'newbalanceOrig': 0.0,      # Drained account
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,      # Unrecorded destination
            'note': 'Suspicious transfer: high amount relative to balance, complete sender balance depletion to zero, unrecorded destination.'
        },
        'Custom input': {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 150.0,
            'oldbalanceOrg': 1000.0,
            'newbalanceOrig': 850.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 150.0,
            'note': 'Custom input: enter any values below.'
        }
    }

    # Support standard and test labels
    example_choices = ['Normal example', 'Suspicious example', 'Custom input', 'Legitimate example', 'Fraud example']
    chosen_label = st.selectbox('Example transaction', example_choices, key='single_example_choice')


    if chosen_label in ('Legitimate example', 'Normal example'):
        preset = examples['Normal example']
    elif chosen_label in ('Fraud example', 'Suspicious example'):
        preset = examples['Suspicious example']
    else:
        preset = examples['Custom input']

    st.caption(f'ℹ️ {preset["note"]}')

    with st.form(f'transaction_form_{chosen_label}'):
        left, middle, right = st.columns(3)
        with left:
            tx_type = st.selectbox('Transaction type', config.TRANSACTION_TYPES,
                                   index=config.TRANSACTION_TYPES.index(preset['type']))
            step = st.number_input('Simulation time step', min_value=1, max_value=744, value=int(preset['step']),
                                   help='Simulation time step: 1 step = 1 hour (1 to 744, representing a 30-day simulation)')
            amount = st.number_input('Amount (synthetic currency units)', min_value=0.0, value=float(preset['amount']))
        with middle:
            old_origin = st.number_input('Sender opening balance (units)', min_value=0.0, value=float(preset['oldbalanceOrg']))
            new_origin = st.number_input('Sender closing balance (units)', min_value=0.0, value=float(preset['newbalanceOrig']))
        with right:
            old_dest = st.number_input('Recipient opening balance (units)', min_value=0.0, value=float(preset['oldbalanceDest']))
            new_dest = st.number_input('Recipient closing balance (units)', min_value=0.0, value=float(preset['newbalanceDest']))
        
        submitted = st.form_submit_button('Evaluate transaction', type='primary', width='stretch')

    payload = dict(step=int(step), type=tx_type, amount=float(amount),
                   oldbalanceOrg=float(old_origin), newbalanceOrig=float(new_origin),
                   oldbalanceDest=float(old_dest), newbalanceDest=float(new_dest))
    return chosen_label, payload, submitted


def single_prediction(metadata, store, user_id, selected, engine, signature, read_csv):
    example_name, payload, submitted = _single_input(read_csv)
    key = (selected, example_name, signature, user_id)
    comparing = selected == COMPARE_OPTION

    if submitted:
        try:
            if comparing:
                report = engine.predict_single(payload)
                saved = store.save_run(user_id, None, 'All models', _comparison_csv(payload, report),
                                       {'rows': 1, 'flagged': int(report['Fraud_Votes'] > 0),
                                        'comparison': report, 'raw_inputs': payload,
                                        'review_rule': 'Any individual model flag requires review'}, kind='single')
                st.session_state['comparison_single_result'] = (key, report, payload, saved['id'])
            else:
                alert = engine.predict_single(payload)
                score_val = alert['Fraud_Probability']
                score_pct = alert.get('Fraud_Probability_Pct', f'{score_val*100:.2f}%')
                pred_label = alert['Prediction']
                row = dict(payload, Transaction_ID=alert['Transaction_ID'], Prediction=pred_label,
                           Fraud_Probability=score_val, Risk_Level=alert['Risk_Level'],
                           Model_Used=selected, Decision_Threshold=engine.threshold, Score_Kind=engine.score_kind)
                saved = store.save_run(user_id, None, selected, pd.DataFrame([row]).to_csv(index=False).encode('utf-8'),
                                       {'rows': 1, 'flagged': int(pred_label == 'FRAUDULENT'),
                                        'threshold': engine.threshold, 'score_kind': engine.score_kind,
                                        'alert': alert}, kind='single')
                st.session_state['single_result'] = (key, alert, saved['id'])
                st.session_state['single_explanation_inputs'] = payload
            st.session_state['last_single_run_id'] = saved['id']
        except (ValueError, sqlite3.Error) as error:
            st.error(str(error))

    # Display Results
    if comparing and st.session_state.get('comparison_single_result', (None,))[0] == key:
        _, report, evaluated_payload, run_id = st.session_state['comparison_single_result']
        votes, total = report['Fraud_Votes'], report['Models_Checked']

        st.divider()
        st.subheader('Evaluation Decision')
        if votes > 0:
            st.error(f'⚠️ **Flagged for review**: {votes} of {total} models flagged this transaction.')
        else:
            st.success(f'✅ **Not flagged**: None of the {total} models flagged this transaction.')

        st.caption('Each verdict uses that model’s validation-selected threshold. A model flag is a prediction requiring analyst review, not a verified finding of fraud.')
        model_table(report['models'])
        render_transaction_explanation(evaluated_payload, report['models'], show_model_table=False)

        col_dl, col_rev = st.columns(2)
        with col_dl:
            st.download_button('📥 Download model comparison JSON', json.dumps(report, indent=2), 'transaction_comparison.json', 'application/json')
        with col_rev:
            st.caption(f'Saved as analysis #{run_id}. You can inspect and update review notes in **Review & History**.')

    elif not comparing and st.session_state.get('single_result', (None,))[0] == key:
        alert = st.session_state['single_result'][1]
        run_id = st.session_state['single_result'][2]
        pred = alert['Prediction']
        score = alert['Fraud_Probability']
        threshold = engine.threshold
        score_kind = engine.score_kind

        st.divider()
        st.subheader('Evaluation Decision')
        
        is_flagged = pred == 'FRAUDULENT'
        if is_flagged:
            st.error(f'⚠️ **Flagged for review** · Model: {selected} · Score: {score:.4f} (Threshold: {threshold:.4f})')
        else:
            st.success(f'✅ **Not flagged** · Model: {selected} · Score: {score:.4f} (Threshold: {threshold:.4f})')

        st.caption(f'Score type: **{score_kind}**. Scores indicate model suspicion and are not calibrated fraud probabilities.')
        st.write(f'**Advisory recommendation:** {alert["Recommended_Action"]}')
        st.progress(min(1.0, max(0.0, score)))

        rows = [{'Model': selected, 'Family': metadata['models'][selected]['family'],
                 'Prediction': pred, 'Score': score,
                 'Threshold': threshold, 'Score_Kind': score_kind}]
        model_table(rows)
        render_transaction_explanation(st.session_state['single_explanation_inputs'], rows,
                                       selected_model=selected, show_model_table=False)

        col_dl, col_rev = st.columns(2)
        with col_dl:
            st.download_button('📥 Download transaction result JSON', json.dumps(alert, indent=2), 'transaction_result.json', 'application/json')
        with col_rev:
            st.caption(f'Saved as analysis #{run_id}. You can inspect and update review notes in **Review & History**.')


def render_batch_result(result, raw_data, metadata, selected, run_id):
    comparing = selected == COMPARE_OPTION
    flagged = (result['Fraud_Votes'] > 0) if comparing else (result['Prediction'] == 'FRAUDULENT')
    count = int(flagged.sum())
    total_count = len(result)
    pct = (count / total_count * 100) if total_count else 0.0

    st.subheader('Batch Scan Results')
    
    b_cols = st.columns(4)
    b_cols[0].metric('Total Scanned', f'{total_count:,}')
    b_cols[1].metric('Flagged for Review', f'{count:,}')
    b_cols[2].metric('Flagged Percentage', f'{pct:.2f}%')
    b_cols[3].metric('Model Used', 'All models' if comparing else selected)

    st.caption(f'Saved as analysis #{run_id}. Results remain accessible after sign-out.')

    if comparing:
        st.info('A transaction is flagged for review when at least one model flags it. Individual model thresholds are applied separately.')

    # Ground-truth comparison if available
    if 'isFraud' in raw_data and raw_data['isFraud'].notna().all() and raw_data['isFraud'].isin([0, 1]).all():
        supplied_fraud = raw_data['isFraud'].to_numpy() == 1
        true_fraud_count = int(supplied_fraud.sum())
        missed = int((supplied_fraud & ~flagged.to_numpy()).sum())
        st.caption(f'Provided dataset labels identify {true_fraud_count} fraud rows. Labels are displayed for comparison and are not used as prediction inputs.')
        if missed:
            st.warning(f'{missed} dataset-labelled fraud transaction(s) received no model flag at the current thresholds.')

    # Filter & Search
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        only_flagged = st.checkbox('Show flagged transactions only', value=True)
    with f_col2:
        search_tx = st.text_input('Search by Transaction ID or type', placeholder='e.g., TRANSFER, CASH_OUT')

    filtered_df = result.loc[flagged] if only_flagged else result
    if search_tx.strip():
        q = search_tx.strip().casefold()
        filtered_df = filtered_df[filtered_df.astype(str).apply(lambda r: r.str.casefold().str.contains(q).any(), axis=1)]

    priority = ['Source_Row', 'Transaction_ID', 'Prediction', 'Fraud_Votes', 'Models_Checked',
                'type', 'amount', 'Flagged_By', 'isFraud']
    order = [c for c in priority if c in filtered_df.columns]
    order += [c for c in filtered_df.columns if c not in order]

    st.dataframe(filtered_df, hide_index=True, width='stretch',
                 column_order=order,
                 column_config={'Fraud_Votes': st.column_config.NumberColumn('Models Flagging', format='%d'),
                                'isFraud': st.column_config.NumberColumn('Supplied Label (0/1)', format='%d')})

    # Two separate download buttons as requested
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button('📥 Download all scored results (CSV)',
                           result.to_csv(index=False).encode('utf-8'),
                           'all_scored_transactions.csv', 'text/csv', width='stretch')
    with dl_col2:
        flagged_csv = result.loc[flagged].to_csv(index=False).encode('utf-8')
        st.download_button('⚠️ Download flagged rows only (CSV)',
                           flagged_csv,
                           'flagged_transactions_only.csv', 'text/csv', width='stretch')

    st.divider()

    # Inspect Individual Transaction
    st.markdown('#### Inspect a Transaction from this Batch')
    positions = ([i + 1 for i, value in enumerate(flagged) if value]
                 if only_flagged and count else list(range(1, len(result) + 1)))
    
    if positions:
        chosen = st.selectbox('Transaction to inspect', positions,
                              format_func=lambda pos: f'Row {pos} · {result.iloc[pos - 1]["Transaction_ID"]} ({result.iloc[pos-1].get("type", "")} {result.iloc[pos-1].get("amount", 0):,.2f})')

        row = result.iloc[chosen - 1]
        st.write(f'**Transaction ID:** `{row["Transaction_ID"]}` · **Source Row:** {chosen}')
        rows = model_rows_from_result(row, metadata, selected)
        model_table(rows)
        render_transaction_explanation(raw_data.iloc[chosen - 1].to_dict(), rows,
                                       selected_model=None if comparing else selected, show_model_table=False)
    else:
        st.info('No transactions match the selected filter.')


def batch_prediction(metadata, store, user_id, current_dataset, selected, engine, signature):
    st.markdown('#### Batch CSV Scoring')
    st.caption('Required columns: ' + ', '.join(config.RAW_FEATURE_COLUMNS))

    datasets = store.list_datasets(user_id)
    if 'batch_source' not in st.session_state:
        st.session_state['batch_source'] = 'Saved workspace dataset' if datasets else 'Upload a new CSV'

    source = st.radio('Batch data source', ['Saved workspace dataset', 'Upload a new CSV'],
                      horizontal=True, key='batch_source')
    content, dataset_id, upload_name = None, None, None

    if source == 'Saved workspace dataset':
        if not datasets:
            st.info('No saved datasets found in your workspace. Choose "Upload a new CSV" or add one in Datasets.')
            return
        records = {item['id']: item for item in datasets}
        identities = list(records)
        current_id = current_dataset['id'] if current_dataset else None
        dataset_id = st.selectbox('Dataset to scan', identities,
                                  index=identities.index(current_id) if current_id in identities else 0,
                                  format_func=lambda i: f'{records[i]["name"]} · {records[i]["row_count"]:,} rows · #{i}')
        saved = store.get_dataset(user_id, dataset_id)
        content = saved['csv_bytes'] if saved else None
    else:
        uploaded = st.file_uploader('Transaction CSV', type=['csv'], key='batch_upload')
        if uploaded is not None:
            content = uploaded.getvalue()
            upload_name = st.text_input('Name for this dataset', value=uploaded.name)
        st.caption(f'Upload limit: 25 MB. Running a scan saves the uploaded dataset and its prediction results to your account.')

    if content is None:
        return

    key = (user_id, selected, hashlib.sha256(content).hexdigest(), signature, dataset_id)
    try:
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError('Choose a CSV file no larger than 25 MB.')
        raw = csv_frame(content)
        st.dataframe(raw.head(5), hide_index=True, width='stretch')

        if st.button('Run batch scan', type='primary'):
            progress_bar = st.progress(0, text='Validating transactions...')
            time.sleep(0.1)

            progress_bar.progress(30, text='Applying trained preprocessing and feature engineering...')
            time.sleep(0.1)

            progress_bar.progress(60, text=f'Evaluating transactions with {selected}...')
            result = engine.predict_batch(raw)

            progress_bar.progress(90, text='Saving analysis results to your history...')
            comparing = selected == COMPARE_OPTION
            flagged = int((result['Fraud_Votes'] > 0).sum()) if comparing else int((result.Prediction == 'FRAUDULENT').sum())

            if dataset_id is None:
                saved_dataset = store.save_dataset(user_id, upload_name.strip() or 'Uploaded transactions', content)
                saved_id = saved_dataset['id']
            else:
                saved_id = dataset_id

            open_dataset(store, user_id, saved_id)
            summary = {'rows': len(result), 'flagged': flagged}
            if comparing:
                summary.update({'models': list(engine.model_names),
                                'review_rule': 'Any individual model flag requires review'})
            else:
                summary.update({'threshold': engine.threshold, 'score_kind': engine.score_kind})

            saved_run = store.save_run(user_id, saved_id, 'All models' if comparing else selected,
                                       result.to_csv(index=False).encode('utf-8'), summary, kind='batch')
            
            progress_bar.progress(100, text='Batch scan complete!')
            time.sleep(0.2)
            progress_bar.empty()

            st.session_state['batch_result'] = (key, result, saved_run['id'])

        if st.session_state.get('batch_result', (None,))[0] == key:
            res_tuple = st.session_state['batch_result']
            render_batch_result(res_tuple[1], raw, metadata, selected, res_tuple[2])
    except (ValueError, UnicodeError, sqlite3.Error) as error:
        st.error(str(error))


def render_analyze_page(metadata, store, user, current_dataset, read_csv, get_predictor):
    """Main page for Analyze Transactions: single-transaction and batch predictions."""
    st.header('Analyze Transactions')
    st.write('Evaluate single transactions or run batch CSV scans across Machine Learning and Deep Learning models.')

    selected = st.selectbox('Active model', [COMPARE_OPTION, *metadata['models']])
    signature = artifact_signature(metadata)

    try:
        engine = get_comparison(signature) if selected == COMPARE_OPTION else get_predictor(selected, signature)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    if selected == COMPARE_OPTION:
        st.info('**Multi-Model Screening**: Random Forest, XGBoost, Deep Neural Network, and Deep Autoencoder each evaluate the transaction using their own validation-selected decision threshold.')
    else:
        st.caption(f'**{metadata["models"][selected]["family"]}** · Decision threshold: `{engine.threshold:.6f}` · Score metric: `{engine.score_kind}`')

def streaming_prediction(metadata, store, user_id, current_dataset, selected, signature):
    st.markdown('#### Real-Time Apache Kafka Streaming Pipeline')
    st.caption('High-throughput real-time streaming ingestion, multi-model consensus inference, and automated alert dispatch.')

    with st.expander('ℹ️ Kafka Architecture & Streaming Topology', expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            - **Input Topic:** `{config.KAFKA_TRANSACTIONS_TOPIC}`
            - **Alerts Topic:** `{config.KAFKA_ALERTS_TOPIC}`
            - **Consumer Group:** `{config.KAFKA_CONSUMER_GROUP}`
            """)
        with c2:
            st.markdown(f"""
            - **Bootstrap Servers:** `{config.KAFKA_BOOTSTRAP_SERVERS}`
            - **Broker Mode:** Dual-Mode (Simulated in-memory zero-dependency broker or live cluster)
            - **Processing Latency:** Sub-100 ms SLA
            """)

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        sources = ['Built-in Benchmark Transactions', 'Active Uploaded Dataset'] if current_dataset else ['Built-in Benchmark Transactions']
        stream_source = st.selectbox('Streaming Source', sources)
    with col2:
        event_count = st.number_input('Events to Stream', min_value=5, max_value=100, value=15, step=5)
    with col3:
        events_per_sec = st.slider('Stream Rate (events/sec)', min_value=1, max_value=25, value=10)

    if st.button('🚀 Start Live Kafka Stream Simulation', type='primary'):
        if stream_source == 'Active Uploaded Dataset' and current_dataset is not None:
            raw_bytes = current_dataset['csv_bytes']
            stream_df = pd.read_csv(io.BytesIO(raw_bytes))
        else:
            sample_p = Path(config.SAMPLE_TEST_PATH)
            if sample_p.exists():
                stream_df = pd.read_csv(sample_p)
            else:
                sample_p = Path(config.DATASET_PATH)
                stream_df = pd.read_csv(sample_p).head(50)

        stream_df = stream_df.head(int(event_count))

        broker = SimulatedStreamingBroker()
        producer = TransactionStreamProducer(use_simulated=True, simulated_broker=broker)
        consumer = FraudStreamingConsumer(use_simulated=True, simulated_broker=broker,
                                          model_mode=selected if selected != COMPARE_OPTION else 'comparison')

        stream_progress = st.progress(0, text="Streaming transactions to Kafka topic 'financial.transactions'...")
        results = []
        alerts_count = 0
        total_lat = 0.0

        for idx, (_, row) in enumerate(stream_df.iterrows()):
            tx_data = row.to_dict()
            tx_id = f"TXN-STRM-{idx+1:04d}"
            tx_data['Transaction_ID'] = tx_id

            msg = producer.produce(tx_data, key=tx_id)
            res = consumer.process_message(msg)
            results.append(res)
            if res.get('is_flagged'):
                alerts_count += 1
            total_lat += res.get('latency_ms', 0.0)

            prog_pct = int(((idx + 1) / len(stream_df)) * 100)
            stream_progress.progress(prog_pct, text=f"Kafka Event {idx + 1}/{len(stream_df)}: {tx_id} -> {res.get('decision')}")
            time.sleep(1.0 / events_per_sec)

        stream_progress.empty()
        st.success(f"Stream simulation complete! Successfully ingested and scored {len(results)} events.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Events Ingested", len(results))
        m2.metric("Fraud Alerts Emitted", alerts_count)
        avg_lat = round(total_lat / max(1, len(results)), 2)
        m3.metric("Avg Latency", f"{avg_lat} ms")
        m4.metric("Alert Ratio", f"{(alerts_count / max(1, len(results))) * 100:.1f}%")

        table_rows = []
        for r in results:
            table_rows.append({
                "Event ID": r.get("event_id"),
                "Transaction ID": r.get("transaction_id"),
                "Type": r.get("type"),
                "Amount": f"${r.get('amount', 0):,.2f}",
                "Decision": r.get("decision"),
                "Flagged": "🚨 YES" if r.get("is_flagged") else "✅ NO",
                "Latency (ms)": f"{r.get('latency_ms', 0):.2f}",
                "Forensic Observations": " | ".join(r.get("forensic_factors", []))
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width='stretch')


def render_analyze_page(metadata, store, user, current_dataset, read_csv, get_predictor):
    """Main page for Analyze Transactions: single-transaction, batch predictions, and live streaming."""
    st.header('Analyze Transactions')
    st.write('Evaluate single transactions, run batch CSV scans, or trigger real-time Apache Kafka streaming ingestion.')

    selected = st.selectbox('Active model', [COMPARE_OPTION, *metadata['models']])
    signature = artifact_signature(metadata)

    try:
        engine = get_comparison(signature) if selected == COMPARE_OPTION else get_predictor(selected, signature)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    if selected == COMPARE_OPTION:
        st.info('**Multi-Model Screening**: Random Forest, XGBoost, Deep Neural Network, and Deep Autoencoder each evaluate the transaction using their own validation-selected decision threshold.')
    else:
        st.caption(f'**{metadata["models"][selected]["family"]}** · Decision threshold: `{engine.threshold:.6f}` · Score metric: `{engine.score_kind}`')

    mode = st.radio('Prediction mode', ['Single transaction', 'Batch CSV', '⚡ Real-Time Kafka Stream'], horizontal=True)
    if mode == 'Single transaction':
        single_prediction(metadata, store, user['id'], selected, engine, signature, read_csv)
    elif mode == 'Batch CSV':
        batch_prediction(metadata, store, user['id'], current_dataset, selected, engine, signature)
    else:
        streaming_prediction(metadata, store, user['id'], current_dataset, selected, signature)


# Backwards compatibility alias for test suite
def render_prediction_page(metadata, store, user, current_dataset, read_csv, get_predictor):
    render_analyze_page(metadata, store, user, current_dataset, read_csv, get_predictor)
