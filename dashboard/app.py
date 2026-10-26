"""FraudGuard AI: Analyst Workspace for Financial Fraud Detection."""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

import config
from predictor import FraudPredictor
from src.workspace_store import WorkspaceStore
from dashboard.auth_page import require_sign_in, sign_out
from dashboard.about_page import render_about_page
from dashboard.workspace_page import active_dataset, csv_frame, open_dataset, render_datasets_page, render_workspace
from dashboard.prediction_page import render_analyze_page, render_prediction_page
from dashboard.review_page import render_review_page
from dashboard.evaluation_page import render_evaluation_page

st.set_page_config(page_title='FraudGuard AI | Analyst Workspace', page_icon='🛡️', layout='wide')

# Custom Modern Styling
st.markdown('''<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
    [data-testid="stMetric"] {background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px;}
    .top-bar {
        background: linear-gradient(90deg, #0f2746 0%, #1e3a8a 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-bar-title {font-size: 1.25rem; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 8px;}
    .demo-badge {
        background: #1e40af;
        color: #dbeafe;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #3b82f6;
    }
    .quick-action-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.15s ease-in-out;
    }
    .stButton>button {border-radius: 8px; font-weight: 600;}
</style>''', unsafe_allow_html=True)

# Compact Header with Demo Data Badge
st.markdown('''
<div class="top-bar">
    <div class="top-bar-title">🛡️ FraudGuard AI <span style="font-size: 0.95rem; font-weight: 400; opacity: 0.85;">| Analyst Workspace</span></div>
    <div class="demo-badge">Synthetic Demo Data</div>
</div>
''', unsafe_allow_html=True)


@st.cache_data
def load_csv(path, modified):
    return pd.read_csv(path)


def read_csv(path):
    return load_csv(str(path), os.path.getmtime(path))


def read_metadata():
    if os.path.exists(config.METADATA_PATH):
        with open(config.METADATA_PATH, encoding='utf-8') as handle:
            return json.load(handle)
    return None


@st.cache_resource
def get_predictor(name, artifact_signature):
    return FraudPredictor(name)


def render_dashboard_page(store, user, current_dataset, runs, metadata):
    """Dashboard: Active dataset, summary counts, latest analysis status, quick actions."""
    st.header('Analyst Dashboard')
    st.write(f'Welcome, **{user["display_name"]}**. Review summary statistics and manage your fraud investigations.')

    user_id = user['id']
    latest_run = runs[0] if runs else None
    
    # 1. Active Dataset & Latest Analysis Cards
    col_data, col_run = st.columns(2, gap='large')

    with col_data:
        with st.container(border=True):
            st.markdown('#### 📁 Active Dataset')
            if current_dataset:
                d_df = csv_frame(current_dataset['csv_bytes'])
                st.write(f'**Name:** {current_dataset["name"]}')
                st.write(f'**Transactions:** {current_dataset["row_count"]:,} rows')
                st.write(f'**Uploaded:** {current_dataset["created_at"][:19]}')
                st.caption('Uploaded data runs against existing models; it does not trigger retraining.')
            else:
                st.info('No active dataset selected yet.')
                st.caption('Upload a CSV or load the demo dataset in the Datasets section.')

    with col_run:
        with st.container(border=True):
            st.markdown('#### ⚡ Latest Prediction Analysis')
            if latest_run:
                l_summary = latest_run.get('summary', {})
                l_rows = l_summary.get('rows', 0)
                l_flagged = l_summary.get('flagged', 0)
                l_pct = (l_flagged / l_rows * 100) if l_rows else 0.0

                r_c1, r_c2, r_c3 = st.columns(3)
                r_c1.metric('Scanned Rows', f'{l_rows:,}')
                r_c2.metric('Flagged', f'{l_flagged:,}')
                r_c3.metric('Flagged %', f'{l_pct:.2f}%')
                st.write(f'**Model Used:** {latest_run["model_name"]} ({latest_run["kind"].title()})')
                st.write(f'**Executed:** {latest_run["created_at"][:19]}')
            else:
                st.info('**Not analyzed yet**')
                st.caption('No predictions have been executed for this account yet. Run an analysis from Analyze Transactions.')

    st.divider()

    # 2. Quick Navigation Actions
    st.subheader('Quick Actions')
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button('📂 Manage Datasets', width='stretch'):
            st.session_state['navigation'] = 'Datasets'
            st.rerun()
        st.caption('Upload, validate and switch datasets.')
    with a2:
        if st.button('🔍 Analyze Transactions', width='stretch'):
            st.session_state['navigation'] = 'Analyze Transactions'
            st.rerun()
        st.caption('Run single or batch model predictions.')
    with a3:
        if st.button('📋 Review Queue', width='stretch'):
            st.session_state['navigation'] = 'Review & History'
            st.rerun()
        st.caption('Inspect flags, add notes, and set status.')
    with a4:
        if st.button('📊 Model Evaluation', width='stretch'):
            st.session_state['navigation'] = 'Model Evaluation'
            st.rerun()
        st.caption('Inspect test benchmark metrics and curves.')

    st.divider()

    # 3. Recent Analyses History Table
    st.subheader('Recent Saved Analyses')
    if runs:
        recent_records = []
        for r in runs[:5]:
            s = r.get('summary', {})
            recent_records.append({
                'ID': f'#{r["id"]}',
                'Model': r['model_name'],
                'Type': r['kind'].title(),
                'Rows': f'{s.get("rows", 0):,}',
                'Flagged': f'{s.get("flagged", 0):,}',
                'Saved (UTC)': r['created_at'][:19]
            })
        st.dataframe(pd.DataFrame(recent_records), hide_index=True, width='stretch')
        st.caption('Open **Review & History** to inspect individual transactions or download full scored datasets.')
    else:
        st.info('No saved analyses in history yet.')

    # Benchmark vs Active Dataset Notice
    st.divider()
    st.caption('📌 **Dataset Distinction**: '
               'Active workspace transactions reflect your uploaded data or demo CSV. '
               'Evaluation metrics on the **Model Evaluation** page reflect the fixed 15,000-row synthetic benchmark experiment.')


def render_app():
    database_path = getattr(config, 'WORKSPACE_DB_PATH', str(BASE_DIR / 'storage' / 'workspaces.sqlite3'))
    with WorkspaceStore(database_path) as store:
        user = require_sign_in(store)
        user_id = user['id']
        metadata = read_metadata()

        # Sidebar Configuration
        st.sidebar.title('FraudGuard AI')
        st.sidebar.markdown(f'**Analyst Workspace**  \n👤 **{user["display_name"]}**  \n📧 `{user["email"]}`  \n🛡️ **Role: Analyst**')
        
        if st.sidebar.button('Sign out', width='stretch'):
            sign_out()

        # Primary 6-Section Navigation
        PAGE_SECTIONS = [
            'Dashboard',
            'Datasets',
            'Analyze Transactions',
            'Review & History',
            'Model Evaluation',
            'About'
        ]

        # Handle page alias mapping (for test suite or session resumption)
        current_nav = st.session_state.get('navigation', 'Dashboard')
        NAV_MAP = {
            'Overview': 'Dashboard',
            'My Workspace': 'Datasets',
            'Data Analysis': 'Datasets',
            'Fraud Prediction & Alerts': 'Analyze Transactions',
            'Model Performance': 'Model Evaluation',
            'Deep Learning': 'Model Evaluation',
            'About & Architecture': 'About'
        }
        if current_nav in NAV_MAP:
            current_nav = NAV_MAP[current_nav]
        if current_nav not in PAGE_SECTIONS:
            current_nav = 'Dashboard'

        page = st.sidebar.radio('Navigation', PAGE_SECTIONS, index=PAGE_SECTIONS.index(current_nav))
        st.session_state['navigation'] = page
        
        st.sidebar.divider()
        st.sidebar.caption('**Models Available:**\n- ML: Random Forest · XGBoost\n- DL: Deep Neural Network · Deep Autoencoder')
        st.sidebar.info('Academic demonstration using synthetic data. Model scores support review; they do not execute financial transactions.')

        if not metadata:
            st.warning('Trained models and metadata not found. Train models to get started.')
            st.code(r'.\.venv\Scripts\python.exe main.py', language='powershell')
            st.stop()

        current_dataset = active_dataset(store, user_id)
        runs = store.list_runs(user_id)
        dataset_name = current_dataset['name'] if current_dataset else 'Synthetic benchmark'
        st.sidebar.caption(f'📁 Active Data: **{dataset_name}**')

        # Route to appropriate page
        if page == 'Dashboard':
            render_dashboard_page(store, user, current_dataset, runs, metadata)

        elif page == 'Datasets':
            render_datasets_page(store, user, current_dataset)

        elif page == 'Analyze Transactions':
            render_analyze_page(metadata, store, user, current_dataset, read_csv, get_predictor)

        elif page == 'Review & History':
            render_review_page(store, user, metadata)

        elif page == 'Model Evaluation':
            render_evaluation_page(metadata, read_csv, current_dataset, store, user_id)


        else:
            render_about_page(metadata)
            st.download_button('📥 Download experiment metadata (JSON)', json.dumps(metadata, indent=2),
                               'experiment_metadata.json', 'application/json')


if __name__ == '__main__':
    render_app()
