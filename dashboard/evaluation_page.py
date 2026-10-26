"""Model Evaluation: Dynamic dataset evaluation, Live confusion matrix & ROC/PR diagrams, Reset state, and Deep Learning."""

import io
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, average_precision_score,
    roc_curve, precision_recall_curve
)

import config
from src.model_comparison import FraudModelComparison
from dashboard.workspace_page import csv_frame, open_dataset

MODEL_COLORS = {
    'Random Forest': '#2563eb',        # Primary Blue
    'XGBoost': '#d97706',              # Warm Orange / Amber
    'Deep Neural Network': '#7c3aed',  # Purple
    'Deep Autoencoder': '#059669',     # Emerald Green
}

MODEL_CMAPS = {
    'Random Forest': 'Blues',
    'XGBoost': 'Oranges',
    'Deep Neural Network': 'Purples',
    'Deep Autoencoder': 'Greens',
}


def fig_to_bytes(fig):
    """Convert matplotlib figure to PNG byte buffer for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    return buf.getvalue()


def show_image(filename, caption=None):
    path = Path(config.RESULTS_DIR) / filename
    if path.exists():
        st.image(str(path), caption=caption, width='stretch')
    else:
        st.caption(f'Chart `{filename}` is not available yet. Train models to generate it.')


@st.cache_resource
def get_comparison_engine():
    return FraudModelComparison(config.MODELS_DIR)


def evaluate_dataset(df):
    """Evaluate all 4 models on a given dataframe and compute metrics if labels exist."""
    comp = get_comparison_engine()
    # Filter out collision columns if already scored
    reserved = ['Source_Row', 'Fraud_Votes', 'Models_Checked', 'Flagged_By', 'Prediction', 'Review_Status']
    clean_cols = [c for c in df.columns if c not in reserved and not any(c.startswith(f'{m} ') for m in config.MODEL_FILES)]
    clean_df = df[clean_cols].copy()
    
    scored = comp.predict_batch(clean_df)
    has_labels = ('isFraud' in df.columns and 
                  df['isFraud'].dropna().isin([0, 1]).all() and 
                  len(df['isFraud'].dropna().unique()) > 0)
    
    model_records = []
    cm_dict = {}
    eval_data = {}

    for name in comp.model_names:
        pred_series = scored[f'{name} Prediction'] == 'FRAUDULENT'
        y_pred = pred_series.astype(int).to_numpy()
        y_score = scored[f'{name} Score'].astype(float).to_numpy()
        threshold = comp._predictors[name].threshold
        score_kind = comp._predictors[name].score_kind

        record = {
            'Model': name,
            'Family': config.MODEL_FAMILIES.get(name, 'Unknown'),
            'Threshold': float(threshold),
            'Score Type': score_kind,
            'Flagged Count': int(pred_series.sum()),
            'Flagged %': f'{(pred_series.sum() / len(df) * 100):.2f}%'
        }

        y_true = None
        if has_labels:
            y_true = df['isFraud'].astype(int).to_numpy()
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            cm_dict[name] = cm

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            try:
                ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
            except Exception:
                ap = 0.0
            try:
                roc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
            except Exception:
                roc = 0.5

            record.update({
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'Average Precision (AP)': ap,
                'ROC-AUC': roc,
                'Accuracy': acc,
                'Specificity': spec,
                'True Positives (TP)': int(tp),
                'False Positives (FP)': int(fp),
                'True Negatives (TN)': int(tn),
                'False Negatives (FN)': int(fn),
            })

        model_records.append(record)
        eval_data[name] = {
            'cm': cm_dict.get(name),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score,
            'threshold': float(threshold),
            'score_kind': score_kind,
            'metrics': record
        }

    return scored, pd.DataFrame(model_records), has_labels, cm_dict, eval_data


def plot_dynamic_confusion_matrices(cm_dict, metrics_df, total_rows):
    """
    Generate dynamic 2x2 side-by-side heatmaps of confusion matrices for all 4 models
    evaluated on the uploaded CSV dataset.
    """
    n_models = len(cm_dict)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 4.9 * rows), dpi=180, squeeze=False)
    axes = axes.ravel()

    metrics_map = metrics_df.set_index('Model').to_dict('index') if not metrics_df.empty else {}

    for idx, (m_name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        total_n = int(cm.sum())
        cm_pct = cm.astype('float') / total_n * 100 if total_n > 0 else np.zeros_like(cm)
        
        labels = np.asarray([
            [f"TN: {cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)", f"FP: {cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)"],
            [f"FN: {cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)", f"TP: {cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)"]
        ])
        
        cmap = MODEL_CMAPS.get(m_name, 'Blues')
        sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, cbar=False, ax=ax,
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                    linewidths=1.5, linecolor='#f1f5f9')
        
        meta = metrics_map.get(m_name, {})
        thresh = meta.get('Threshold', 0.5)
        f1_val = meta.get('F1-Score', 0.0)
        
        ax.set_title(f"{m_name}\nDecision Threshold: {thresh:.4f}  |  F1: {f1_val:.4f}",
                     fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight='semibold')
        ax.set_ylabel("Actual Ground Truth", fontsize=10, fontweight='semibold')
        ax.set_xticklabels(['Legitimate (0)', 'Fraud (1)'], fontsize=9.5)
        ax.set_yticklabels(['Legitimate (0)', 'Fraud (1)'], fontsize=9.5)

    for unused in axes[n_models:]:
        unused.set_visible(False)

    plt.suptitle(f"Dynamic Confusion Matrices on Uploaded Dataset (N = {total_rows:,})\n"
                 f"Percentages computed relative to total transactions in uploaded CSV",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_dynamic_roc_pr_curves(eval_data, fraud_rate, total_rows):
    """
    Generate dynamic comparative ROC and Precision-Recall curves on the uploaded CSV dataset.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=180)
    
    # 1. ROC Curves
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.55, linewidth=1.2, label='Random Guess (AUC = 0.5000)')
    
    for m_name, d in eval_data.items():
        y_true = d['y_true']
        y_score = d['y_score']
        color = MODEL_COLORS.get(m_name, '#2563eb')
        if y_true is not None and len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)
            ax_roc.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{m_name} (AUC = {auc_val:.4f})")
        else:
            ax_roc.plot([], [], color=color, label=f"{m_name} (Single class)")

    ax_roc.set_title(f"Comparative ROC Curves (N={total_rows:,})", fontsize=12, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight='semibold')
    ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10, fontweight='semibold')
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    ax_roc.legend(loc='lower right', frameon=True, fontsize=9.5)

    # 2. Precision-Recall Curves
    ax_pr.axhline(y=fraud_rate, color='gray', linestyle='--', linewidth=1.2,
                  label=f'Prevalence Baseline ({fraud_rate:.2%})')
    
    for m_name, d in eval_data.items():
        y_true = d['y_true']
        y_score = d['y_score']
        color = MODEL_COLORS.get(m_name, '#2563eb')
        if y_true is not None and len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap_val = average_precision_score(y_true, y_score)
            ax_pr.plot(rec, prec, color=color, linewidth=2.0, label=f"{m_name} (AP = {ap_val:.4f})")
            
            # Highlight operational threshold point
            m_metrics = d.get('metrics', {})
            op_p = m_metrics.get('Precision', None)
            op_r = m_metrics.get('Recall', None)
            if op_p is not None and op_r is not None:
                ax_pr.scatter([op_r], [op_p], color=color, s=80, zorder=5, edgecolors='black', linewidth=1.2)
        else:
            ax_pr.plot([], [], color=color, label=f"{m_name} (Single class)")

    ax_pr.set_title(f"Comparative Precision-Recall Curves (N={total_rows:,})", fontsize=12, fontweight='bold')
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight='semibold')
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=10, fontweight='semibold')
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    ax_pr.legend(loc='lower left', frameon=True, fontsize=9.5)

    plt.suptitle("Dynamic Operating Tradeoffs on Uploaded Dataset", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_dynamic_dl_distributions(eval_data, total_rows):
    """
    Generate dynamic distribution plots for Deep Neural Network and Deep Autoencoder on uploaded CSV.
    """
    fig, (ax_dnn, ax_ae) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=180)
    
    # 1. DNN Fraud Probability
    dnn_data = eval_data.get('Deep Neural Network')
    if dnn_data:
        scores = dnn_data['y_score']
        thresh = dnn_data['threshold']
        y_true = dnn_data['y_true']
        
        if y_true is not None and len(np.unique(y_true)) > 1:
            sns.histplot(scores[y_true == 0], color='#2563eb', label='Legitimate (0)', bins=25, kde=True, ax=ax_dnn, alpha=0.45)
            sns.histplot(scores[y_true == 1], color='#dc2626', label='Fraud (1)', bins=25, kde=True, ax=ax_dnn, alpha=0.55)
        else:
            sns.histplot(scores, color='#7c3aed', label='Transactions', bins=25, kde=True, ax=ax_dnn, alpha=0.5)
            
        ax_dnn.axvline(thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold ({thresh:.4f})')
        ax_dnn.set_title("Deep Neural Network: Fraud Probability Distribution", fontsize=11.5, fontweight='bold')
        ax_dnn.set_xlabel("Output Sigmoid Probability", fontsize=9.5, fontweight='semibold')
        ax_dnn.set_ylabel("Transaction Count", fontsize=9.5, fontweight='semibold')
        ax_dnn.legend(fontsize=9)
        ax_dnn.grid(True, linestyle=':', alpha=0.6)

    # 2. Autoencoder Reconstruction Error
    ae_data = eval_data.get('Deep Autoencoder')
    if ae_data:
        scores = ae_data['y_score']
        thresh = ae_data['threshold']
        y_true = ae_data['y_true']
        
        if y_true is not None and len(np.unique(y_true)) > 1:
            sns.histplot(scores[y_true == 0], color='#059669', label='Legitimate (0)', bins=25, kde=True, ax=ax_ae, alpha=0.45)
            sns.histplot(scores[y_true == 1], color='#dc2626', label='Fraud (1)', bins=25, kde=True, ax=ax_ae, alpha=0.55)
        else:
            sns.histplot(scores, color='#059669', label='Transactions', bins=25, kde=True, ax=ax_ae, alpha=0.5)
            
        ax_ae.axvline(thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold ({thresh:.4f})')
        ax_ae.set_title("Deep Autoencoder: Reconstruction Error Anomaly Score", fontsize=11.5, fontweight='bold')
        ax_ae.set_xlabel("Normalized Reconstruction Error Score", fontsize=9.5, fontweight='semibold')
        ax_ae.set_ylabel("Transaction Count", fontsize=9.5, fontweight='semibold')
        ax_ae.legend(fontsize=9)
        ax_ae.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"Deep Learning Anomaly & Score Distributions on Uploaded Data (N={total_rows:,})",
                 fontsize=12.5, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_dynamic_agreement_matrix(scored_df, comp_model_names, total_rows):
    """
    Generate dynamic model consensus heatmap showing pairwise flag agreement across all models on unlabelled data.
    """
    agreement = np.zeros((len(comp_model_names), len(comp_model_names)))
    for i, m1 in enumerate(comp_model_names):
        pred1 = scored_df[f'{m1} Prediction'] == 'FRAUDULENT'
        for j, m2 in enumerate(comp_model_names):
            pred2 = scored_df[f'{m2} Prediction'] == 'FRAUDULENT'
            agreement[i, j] = (pred1 == pred2).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    sns.heatmap(agreement, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax,
                xticklabels=comp_model_names, yticklabels=comp_model_names,
                cbar_kws={'label': 'Decision Agreement %'}, annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title(f"Model Pairwise Decision Agreement (%)\non Uploaded Dataset (N={total_rows:,})",
                 fontsize=11.5, fontweight='bold', pad=10)
    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def plot_dynamic_feature_importance():
    """
    Plot feature importance of the gradient-boosted tree ensemble (XGBoost).
    """
    try:
        comp = get_comparison_engine()
        xgb_pred = comp._predictors.get('XGBoost')
        if not xgb_pred:
            return None
        model = xgb_pred.model
        feature_names = xgb_pred.preprocessor.feature_names
        importances = model.feature_importances_
        
        top_k = min(10, len(feature_names))
        idx = np.argsort(importances)[::-1][:top_k]
        top_names = [feature_names[i] for i in idx]
        top_scores = importances[idx]
        
        fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=180)
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_scores[::-1], color='#d97706', alpha=0.85, edgecolor='#78350f', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=9.5)
        ax.set_xlabel("Relative Feature Importance Weight", fontsize=10, fontweight='semibold')
        ax.set_title("XGBoost Top Feature Importances (Tree Explanations)", fontsize=11.5, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5, axis='x')
        plt.tight_layout()
        return fig
    except Exception:
        return None


def render_evaluation_page(metadata, read_csv, current_dataset=None, store=None, user_id=None, *args, **kwargs):
    st.header('Model Evaluation')
    st.caption('Dynamically evaluate all Machine Learning and Deep Learning models on any uploaded CSV data. Diagrams, confusion matrices, and metrics update in real time.')

    # -------------------------------------------------------------
    # 1. DATASET SOURCE SELECTION & UPLOAD (Directly on this page!)
    # -------------------------------------------------------------
    with st.container(border=True):
        st.markdown('#### 📂 Select or Upload Dataset for Dynamic Evaluation')
        
        source_mode = st.radio(
            'Evaluation Data Source',
            ['Upload a new CSV file', 'Active / Saved workspace dataset', 'Quick test benchmarks'],
            horizontal=True,
            key='eval_source_mode'
        )

        active_bytes = None
        dataset_display_name = None

        if source_mode == 'Upload a new CSV file':
            c_up1, c_up2 = st.columns([2, 1])
            with c_up1:
                uploaded_eval_file = st.file_uploader(
                    'Choose a CSV file to evaluate',
                    type=['csv'],
                    key='eval_direct_csv_uploader',
                    help='Upload any transaction CSV file up to 25 MB.'
                )
            with c_up2:
                custom_name = st.text_input('Dataset name', placeholder='e.g., Audit Evaluation Q4', key='eval_upload_name')
                save_to_ws = st.checkbox('Save to workspace datasets', value=True, key='eval_save_cb')

            if uploaded_eval_file is not None:
                active_bytes = uploaded_eval_file.getvalue()
                dataset_display_name = custom_name.strip() or uploaded_eval_file.name
                # Persist in workspace if requested and store available
                if save_to_ws and store and user_id and active_bytes:
                    saved_info = store.save_dataset(user_id, dataset_display_name, active_bytes)
                    open_dataset(store, user_id, saved_info['id'])
                    st.session_state['active_dataset_id'] = saved_info['id']

        elif source_mode == 'Active / Saved workspace dataset':
            saved_list = store.list_datasets(user_id) if store and user_id else []
            if not saved_list:
                st.info('No saved datasets found in your workspace. Choose "Upload a new CSV file" or "Quick test benchmarks".')
            else:
                records = {item['id']: item for item in saved_list}
                identities = list(records)
                current_id = current_dataset['id'] if current_dataset else identities[0]
                selected_id = st.selectbox(
                    'Select saved dataset to evaluate',
                    identities,
                    index=identities.index(current_id) if current_id in identities else 0,
                    format_func=lambda i: f'{records[i]["name"]} ({records[i]["row_count"]:,} rows · #{i})',
                    key='eval_select_saved_ds'
                )
                saved_record = store.get_dataset(user_id, selected_id) if store and user_id else None
                if saved_record:
                    active_bytes = saved_record['csv_bytes']
                    dataset_display_name = saved_record['name']
                    open_dataset(store, user_id, selected_id)
                    st.session_state['active_dataset_id'] = selected_id

        else:  # Quick test benchmarks
            benchmark_options = {
                'Synthetic Sample Test (30 rows, fast demo)': config.SAMPLE_TEST_PATH,
                'Curated Fraud Cases (50 high-risk fraud cases)': 'data/fraud_cases_test.csv',
                'PaySim Sample (1,000 mobile-money rows)': 'data/paysim_sample_1000.csv',
                'Full Benchmark Transactions (15,000 rows)': config.TRANSACTIONS_CSV_PATH,
            }
            chosen_bench = st.selectbox('Choose a pre-packaged benchmark dataset', list(benchmark_options.keys()), key='eval_quick_bench_sel')
            bench_path = Path(benchmark_options[chosen_bench])
            if bench_path.exists():
                active_bytes = bench_path.read_bytes()
                dataset_display_name = chosen_bench.split(' (')[0]
                if st.button('⚡ Set as Active Dataset in Workspace', key='set_bench_active_btn'):
                    if store and user_id:
                        saved_info = store.save_dataset(user_id, dataset_display_name, active_bytes)
                        open_dataset(store, user_id, saved_info['id'])
                        st.session_state['active_dataset_id'] = saved_info['id']
                        st.rerun()

    # -------------------------------------------------------------
    # 2. CHECK IF DATA IS AVAILABLE TO EVALUATE
    # -------------------------------------------------------------
    if active_bytes is None:
        if current_dataset is not None:
            active_bytes = current_dataset['csv_bytes']
            dataset_display_name = current_dataset['name']
        else:
            st.divider()
            st.subheader('Status: Awaiting Dataset')
            st.info('👆 **Upload a CSV file or select a dataset above** to dynamically compute confusion matrices, comparative curves, and model metrics.')
            with st.expander('ℹ️ View Offline Pre-Trained Benchmark Baseline (15,000 Transactions)'):
                render_benchmark_baseline(metadata, read_csv)
            return

    # Parse and validate the dataset dataframe
    try:
        active_df = csv_frame(active_bytes)
    except Exception as e:
        st.error(f'Failed to parse dataset CSV: {e}')
        return

    missing_required = [c for c in config.RAW_FEATURE_COLUMNS if c not in active_df.columns]
    if missing_required:
        st.error(f'Uploaded dataset is missing required columns: {", ".join(missing_required)}. Required columns are: {", ".join(config.RAW_FEATURE_COLUMNS)}')
        return

    # -------------------------------------------------------------
    # 3. DYNAMIC EVALUATION RUN
    # -------------------------------------------------------------
    col_hdr, col_rst = st.columns([3, 1])
    with col_hdr:
        st.markdown(f'### 📊 Evaluation Results: **{dataset_display_name}**')
        st.caption(f'Total Rows: **{len(active_df):,}** · Required feature columns verified.')
    with col_rst:
        if st.button('🔄 Reset State', help='Reset the evaluation view', key='reset_eval_page_btn'):
            if store and user_id:
                store.set_active_dataset(user_id, None)
            st.session_state.pop('active_dataset_id', None)
            st.session_state.pop('eval_direct_csv_uploader', None)
            st.rerun()

    with st.spinner('Scoring uploaded data with all 4 models and computing live metrics...'):
        scored_df, metrics_df, has_labels, cm_dict, eval_data = evaluate_dataset(active_df)

    total_rows = len(active_df)
    fraud_rate = (active_df['isFraud'].sum() / total_rows) if has_labels and total_rows > 0 else 0.0

    # -------------------------------------------------------------
    # 4. TABS: ALL METRICS & DIAGRAMS ADAPT TO UPLOADED CSV
    # -------------------------------------------------------------
    eval_tabs = st.tabs([
        '📈 Active Dataset Evaluation',
        '🎯 Confusion Matrices',
        '📉 Curves & Feature Importance',
        '🧠 Deep Learning',
        '🏛️ Benchmark Baseline'
    ])

    # -------------------------------------------------------------
    # TAB 1: ACTIVE DATASET EVALUATION TABLE & METRICS
    # -------------------------------------------------------------
    with eval_tabs[0]:
        st.subheader(f'Performance Summary: {dataset_display_name}')

        if has_labels:
            total_fraud = int(active_df['isFraud'].sum())
            fraud_pct = (total_fraud / total_rows * 100) if total_rows else 0.0
            best_row = metrics_df.sort_values(by='F1-Score', ascending=False).iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Total Scanned', f'{total_rows:,}')
            m2.metric('Ground-Truth Fraud', f'{total_fraud:,}')
            m3.metric('Fraud Prevalence', f'{fraud_pct:.2f}%')
            m4.metric('Top Model (by F1)', f'{best_row["Model"]} ({best_row["F1-Score"]:.4f})')

            st.markdown('#### Model Comparison on Uploaded Data')
            compact_cols = ['Model', 'Family', 'Precision', 'Recall', 'F1-Score', 'Average Precision (AP)', 'ROC-AUC']
            compact_display = metrics_df[compact_cols].copy()
            st.dataframe(compact_display, hide_index=True, width='stretch',
                         column_config={
                             'Precision': st.column_config.NumberColumn(format='%.2%'),
                             'Recall': st.column_config.NumberColumn(format='%.2%'),
                             'F1-Score': st.column_config.NumberColumn(format='%.4f'),
                             'Average Precision (AP)': st.column_config.NumberColumn(format='%.4f'),
                             'ROC-AUC': st.column_config.NumberColumn(format='%.4f'),
                         })

            # Plain-language explanation for top model
            st.markdown('**Plain-Language Findings for Top Model:**')
            st.success(
                f'🛡️ **{best_row["Model"]}** detected **{best_row["True Positives (TP)"]} of {total_fraud}** fraud cases '
                f'({best_row["Recall"]:.2%} Recall), missed **{best_row["False Negatives (FN)"]}** fraud cases, and '
                f'generated **{best_row["False Positives (FP)"]}** false alarms out of {total_rows - total_fraud} legitimate transactions '
                f'({best_row["Precision"]:.2%} Precision).'
            )

            # Extended metrics drawer
            with st.expander('View extended confusion counts & accuracy metrics'):
                extended_cols = ['Model', 'Accuracy', 'Specificity', 'True Positives (TP)', 'False Positives (FP)', 'True Negatives (TN)', 'False Negatives (FN)', 'Threshold', 'Score Type']
                st.dataframe(metrics_df[extended_cols], hide_index=True, width='stretch')

            # Performance Bar Chart
            st.markdown('#### Comparative Performance Chart (Uploaded CSV)')
            chart_df = metrics_df.set_index('Model')[['Precision', 'Recall', 'F1-Score']]
            st.bar_chart(chart_df)

            st.download_button('📥 Download evaluation metrics (CSV)',
                               metrics_df.to_csv(index=False).encode(),
                               f'evaluation_{dataset_display_name.replace(" ", "_")}.csv', 'text/csv')

        else:
            st.info('ℹ️ **Unlabelled Dataset**: Ground-truth fraud labels (`isFraud`) were not provided in this CSV. '
                    'Ground-truth evaluation metrics (Precision, Recall, F1) require true labels. '
                    'Showing model flag counts and consensus below:')
            
            u1, u2, u3 = st.columns(3)
            u1.metric('Total Scanned', f'{total_rows:,}')
            any_flagged = int((scored_df['Fraud_Votes'] > 0).sum())
            u2.metric('Flagged by ≥ 1 Model', f'{any_flagged:,}')
            u3.metric('Flagged Share', f'{(any_flagged / total_rows * 100):.2f}%' if total_rows else '0.00%')

            unlabelled_cols = ['Model', 'Family', 'Threshold', 'Score Type', 'Flagged Count', 'Flagged %']
            st.dataframe(metrics_df[unlabelled_cols], hide_index=True, width='stretch')

            st.markdown('#### Model Detection Counts')
            chart_unlab = metrics_df.set_index('Model')['Flagged Count']
            st.bar_chart(chart_unlab)

    # -------------------------------------------------------------
    # TAB 2: DYNAMIC CONFUSION MATRIX DIAGRAMS
    # -------------------------------------------------------------
    with eval_tabs[1]:
        st.subheader('Dynamic Confusion Matrix Diagrams')
        
        if has_labels and cm_dict:
            st.caption(f'Generated dynamically for uploaded dataset: **{dataset_display_name}** (N = {total_rows:,}).')
            
            # Generate the live matplotlib / seaborn figure
            fig_cm = plot_dynamic_confusion_matrices(cm_dict, metrics_df, total_rows)
            st.pyplot(fig_cm)
            
            # Download button for the dynamic diagram
            cm_bytes = fig_to_bytes(fig_cm)
            plt.close(fig_cm)
            
            st.download_button(
                '📥 Download Dynamic Confusion Matrices (PNG)',
                cm_bytes,
                file_name=f'confusion_matrices_{dataset_display_name.replace(" ", "_")}.png',
                mime='image/png',
                key='dl_cm_png_btn'
            )

            st.divider()
            st.markdown('#### Model Breakdown Details')
            cm_cols = st.columns(len(cm_dict))
            for i, (m_name, cm) in enumerate(cm_dict.items()):
                with cm_cols[i % len(cm_cols)]:
                    with st.container(border=True):
                        st.markdown(f'**{m_name}**')
                        tn, fp, fn, tp = cm.ravel()
                        total = cm.sum()
                        st.write(f'✅ **True Negatives:** {tn:,} ({tn/total*100:.1f}%)')
                        st.write(f'⚠️ **False Positives:** {fp:,} ({fp/total*100:.1f}%)')
                        st.write(f'❌ **False Negatives:** {fn:,} ({fn/total*100:.1f}%)')
                        st.write(f'🎯 **True Positives:** {tp:,} ({tp/total*100:.1f}%)')

        else:
            st.info('ℹ️ The uploaded dataset does not contain an `isFraud` column. Confusion matrices require ground-truth labels (0 = Legitimate, 1 = Fraud).')
            st.caption('Displaying pairwise model decision agreement matrix for this uploaded data:')
            comp = get_comparison_engine()
            fig_agree = plot_dynamic_agreement_matrix(scored_df, comp.model_names, total_rows)
            st.pyplot(fig_agree)
            plt.close(fig_agree)

    # -------------------------------------------------------------
    # TAB 3: DYNAMIC ROC & PRECISION-RECALL CURVES & FEATURE IMPORTANCE
    # -------------------------------------------------------------
    with eval_tabs[2]:
        st.subheader('Dynamic Tradeoff Curves & Feature Importance')

        if has_labels:
            st.caption(f'Curves generated dynamically from prediction scores on: **{dataset_display_name}** (N = {total_rows:,}).')
            fig_curves = plot_dynamic_roc_pr_curves(eval_data, fraud_rate, total_rows)
            st.pyplot(fig_curves)
            
            curves_bytes = fig_to_bytes(fig_curves)
            plt.close(fig_curves)
            
            st.download_button(
                '📥 Download Dynamic ROC & PR Curves (PNG)',
                curves_bytes,
                file_name=f'roc_pr_curves_{dataset_display_name.replace(" ", "_")}.png',
                mime='image/png',
                key='dl_curves_png_btn'
            )
        else:
            st.info('Ground-truth `isFraud` labels are required to plot ROC and Precision-Recall curves. Displaying feature importances below:')

        st.divider()
        st.markdown('#### Feature Importance & Tree Explanations')
        fig_feat = plot_dynamic_feature_importance()
        if fig_feat:
            st.pyplot(fig_feat)
            plt.close(fig_feat)
        else:
            show_image('shap_feature_importance.png', caption='Global SHAP Feature Importance (Pre-Trained Benchmark)')

    # -------------------------------------------------------------
    # TAB 4: DEEP LEARNING ARCHITECTURES & SCORE DISTRIBUTIONS
    # -------------------------------------------------------------
    with eval_tabs[3]:
        st.subheader('Deep Learning Models & Anomaly Scoring')
        d1, d2 = st.columns(2)
        with d1:
            with st.container(border=True):
                st.markdown('**Supervised Deep Neural Network (DNN)**')
                st.code('14 inputs → Dense(64, ReLU) → Dense(32, ReLU)\n→ Dense(16, ReLU) → Dense(1, Sigmoid)')
                st.caption('Trained with binary cross-entropy, Adam optimizer, L2 regularization, early stopping.')
        with d2:
            with st.container(border=True):
                st.markdown('**Unsupervised Deep Autoencoder (AE)**')
                st.code('14 inputs → 64 → 32 → 8 (bottleneck)\n→ 32 → 64 → 14 outputs')
                st.caption('Trained on legitimate transactions only with MSE loss. Unusual transactions yield higher reconstruction error.')

        st.divider()
        st.markdown('#### Dynamic Deep Learning Score Distributions on Uploaded Data')
        fig_dl = plot_dynamic_dl_distributions(eval_data, total_rows)
        st.pyplot(fig_dl)
        dl_bytes = fig_to_bytes(fig_dl)
        plt.close(fig_dl)

        st.download_button(
            '📥 Download Deep Learning Distribution Plot (PNG)',
            dl_bytes,
            file_name=f'dl_distributions_{dataset_display_name.replace(" ", "_")}.png',
            mime='image/png',
            key='dl_dist_png_btn'
        )

    # -------------------------------------------------------------
    # TAB 5: OFFLINE PRE-TRAINED BENCHMARK BASELINE
    # -------------------------------------------------------------
    with eval_tabs[4]:
        render_benchmark_baseline(metadata, read_csv)


def render_benchmark_baseline(metadata, read_csv):
    """Render the offline benchmark baseline training and test results."""
    st.subheader('Offline Pre-Trained Benchmark Baseline')
    st.caption('The baseline experiment was trained on 15,000 synthetic benchmark transactions with stratified partitions.')

    test_csv_path = Path(config.COMPARISON_CSV_PATH)
    if test_csv_path.exists():
        test_df = read_csv(test_csv_path)
        compact_cols = ['Model', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'PR-AUC']
        avail = [c for c in compact_cols if c in test_df.columns]
        st.dataframe(test_df[avail], hide_index=True, width='stretch')

    show_image('model_comparison_chart.png', caption='Offline test partition benchmark.')
    show_image('class_distribution_before_after_smote.png', caption='Training set class distribution before and after SMOTE-NC.')
