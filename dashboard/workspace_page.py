"""Account-owned dataset management, validation, preview, and selection."""

import io
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

import config
from src.dataset_validation import MAX_UPLOAD_BYTES, UPLOAD_LIMIT_MB, validate_dataset

MAX_FILE_BYTES = MAX_UPLOAD_BYTES


def csv_frame(csv_bytes):
    return pd.read_csv(io.BytesIO(csv_bytes))


def open_dataset(store, user_id, dataset_id):
    dataset = store.get_dataset(user_id, dataset_id)
    if dataset is None:
        raise ValueError('This dataset is not available in your account.')
    store.set_active_dataset(user_id, dataset_id)
    st.session_state['active_dataset_id'] = dataset_id
    st.session_state.pop('batch_result', None)
    return dataset


def active_dataset(store, user_id):
    dataset_id = st.session_state.get('active_dataset_id')
    if dataset_id:
        dataset = store.get_dataset(user_id, dataset_id)
        if dataset:
            return dataset
        st.session_state.pop('active_dataset_id', None)
    dataset = store.get_active_dataset(user_id)
    if dataset:
        st.session_state['active_dataset_id'] = dataset['id']
    return dataset


def render_saved_analysis(store, user_id, run):
    """Render saved run summary and results (backwards compatible helper)."""
    result = store.get_run(user_id, run['id'])
    if result is None:
        st.info('This saved analysis is unavailable.')
        return
    st.subheader('Saved analysis')
    st.caption(f'{result["model_name"]} · {result["kind"].title()} · {result["created_at"]}')
    summary = result.get('summary', {})
    columns = st.columns(3)
    columns[0].metric('Transactions', f'{summary.get("rows", 0):,}')
    columns[1].metric('Flagged', f'{summary.get("flagged", 0):,}')
    columns[2].metric('Model', result['model_name'])
    st.dataframe(csv_frame(result['result_csv']), hide_index=True, width='stretch')
    st.download_button('Download saved analysis CSV', result['result_csv'],
                       file_name=f'analysis_{result["id"]}.csv', mime='text/csv',
                       key=f'download_run_{result["id"]}')


def render_datasets_page(store, user, current):
    """Datasets section: Upload, validate, preview, select, and manage datasets."""
    user_id = user['id']
    st.header('Datasets')
    st.write(f'Welcome, {user["display_name"]}. Start an analysis or continue reviewing your saved transactions.')

    # Prominent active dataset banner
    if current:
        st.info(f'📁 **Active Dataset:** {current["name"]} ({current["row_count"]:,} transactions). '
                'Predictions run on existing models; uploaded datasets do not retrain models.')
    else:
        st.warning('No active dataset selected. Upload a transaction CSV or try the demo dataset below.')

    datasets = store.list_datasets(user_id)

    # 1. Upload & Demo Section (Grouped Together)
    st.subheader('Add or Load a Dataset')
    st.caption(f'File size limit: {UPLOAD_LIMIT_MB} MB across all tools. Required columns: ' + ', '.join(config.RAW_FEATURE_COLUMNS))

    col_upload, col_actions = st.columns([1.6, 1], gap='medium')

    with col_actions:
        with st.container(border=True):
            st.markdown('**Quick Actions**')
            # Download Sample CSV
            sample_path = Path(config.SAMPLE_TEST_PATH)
            if sample_path.exists():
                st.download_button(
                    '📥 Download sample CSV',
                    sample_path.read_bytes(),
                    file_name='sample_transactions.csv',
                    mime='text/csv',
                    width='stretch',
                    help='Download 30 held-out synthetic test transactions with standard format.'
                )
            
            # Try demo dataset button
            if st.button('Add demo dataset', width='stretch', help='Load 30 synthetic example transactions to your workspace.'):
                if sample_path.exists():
                    saved = store.save_dataset(user_id, 'Synthetic Demo Transactions', sample_path.read_bytes())
                    open_dataset(store, user_id, saved['id'])
                    st.session_state['workspace_message'] = 'Demo dataset loaded and set as active dataset.'
                    st.rerun()

            st.caption('Sample data allows you to test the prediction workflow without creating your own CSV.')

    with col_upload:
        with st.container(border=True):
            st.markdown('**Upload Transaction CSV**')
            uploaded_file = st.file_uploader(
                'Choose a CSV file',
                type=['csv'],
                key='dataset_csv_uploader',
                help=f'Maximum upload size is {UPLOAD_LIMIT_MB} MB.'
            )
            dataset_name_input = st.text_input('Dataset name', placeholder='e.g., Q4 Transaction Audit', key='upload_dataset_name')

            if uploaded_file is not None:
                content = uploaded_file.getvalue()
                if len(content) > MAX_FILE_BYTES:
                    st.error(f'File exceeds the {UPLOAD_LIMIT_MB} MB limit ({len(content) / (1024*1024):.1f} MB). Please choose a smaller file.')
                else:
                    try:
                        preview_df = csv_frame(content)
                        val_result = validate_dataset(preview_df)
                        summary = val_result['summary']

                        st.markdown('**Pre-Upload Validation Preview:**')
                        v_cols = st.columns(4)
                        v_cols[0].metric('Total Rows', f'{summary["total_rows"]:,}')
                        v_cols[1].metric('Valid Rows', f'{summary["valid_rows"]:,}')
                        v_cols[2].metric('Missing Values', f'{summary["missing_values"]:,}')
                        v_cols[3].metric('Duplicate Rows', f'{summary["duplicate_rows"]:,}')

                        if summary['duplicate_rows'] > 0:
                            st.caption(f'ℹ️ {summary["duplicate_rows"]} duplicate transaction rows detected; valid repeated transactions are preserved.')

                        if val_result['missing_columns']:
                            st.error('Missing required columns: ' + ', '.join(val_result['missing_columns']))

                        # Show validation issues if any
                        issues_df = val_result['issues']
                        if not issues_df.empty:
                            with st.expander(f'View {len(issues_df)} validation warnings/issues', expanded=not val_result['can_analyze']):
                                st.dataframe(issues_df.head(50), hide_index=True, width='stretch')

                        st.dataframe(preview_df.head(5), hide_index=True, width='stretch')

                        if val_result['can_analyze']:
                            if st.button('Save and Set as Active Dataset', type='primary'):
                                name = dataset_name_input.strip() or uploaded_file.name
                                saved_d = store.save_dataset(user_id, name, content)
                                open_dataset(store, user_id, saved_d['id'])
                                st.session_state['workspace_message'] = f'Dataset "{name}" successfully saved and activated.'
                                st.rerun()
                        else:
                            st.error('This CSV cannot be analyzed due to validation errors. Please correct the columns and format.')
                    except Exception as exc:
                        st.error(f'Unable to parse CSV file: {exc}')

    if st.session_state.get('workspace_message'):
        st.success(st.session_state.pop('workspace_message'))

    st.divider()

    # 2. Manage & Switch Saved Datasets
    st.subheader('Saved Datasets')
    if datasets:
        records = {item['id']: item for item in datasets}
        choices = list(records)
        current_id = current['id'] if current else None
        
        c_select, c_btn = st.columns([3, 1])
        with c_select:
            chosen = st.selectbox(
                'Saved datasets',
                choices,
                index=choices.index(current_id) if current_id in choices else 0,
                format_func=lambda i: f'{records[i]["name"]} · {records[i]["row_count"]:,} rows · Saved {records[i]["created_at"][:10]}'
            )
        with c_btn:
            st.write('')
            st.write('')
            if st.button('Open dataset', type='primary', width='stretch'):
                open_dataset(store, user_id, chosen)
                st.session_state['workspace_message'] = f'Active dataset switched to "{records[chosen]["name"]}".'
                st.rerun()


        if current:
            st.markdown(f'#### Dataset Inspection: {current["name"]}')
            curr_df = csv_frame(current['csv_bytes'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Transaction Type Breakdown**')
                if 'type' in curr_df.columns:
                    st.bar_chart(curr_df['type'].value_counts())
            with c2:
                st.markdown('**Class Distribution (if labelled)**')
                if 'isFraud' in curr_df.columns and curr_df['isFraud'].dropna().isin([0, 1]).all():
                    st.bar_chart(curr_df['isFraud'].map({0: 'Legitimate (0)', 1: 'Fraud (1)'}).value_counts())
                else:
                    st.info('Dataset does not include ground-truth fraud labels. Run predictions in Analyze Transactions to score rows.')

            st.markdown('**Preview (first 50 rows)**')
            st.dataframe(curr_df.head(50), hide_index=True, width='stretch')
            st.download_button('📥 Download active dataset CSV', current['csv_bytes'],
                               file_name=f'{current["name"].replace(" ", "_")}.csv', mime='text/csv')
    else:
        st.info('No datasets yet. Upload a transaction CSV or try the demo dataset.')

    # 3. Saved Analysis History
    st.divider()
    st.subheader('Analysis history')
    runs = store.list_runs(user_id)
    if runs:
        labels = {item['id']: item for item in runs}
        chosen = st.selectbox('Saved analyses', list(labels), format_func=lambda identity:
                              f'{labels[identity]["model_name"]} · {labels[identity]["kind"]} · {labels[identity]["created_at"]} · #{identity}')
        if st.button('Open saved analysis'):
            st.session_state['saved_run_id'] = chosen
        viewed = st.session_state.get('saved_run_id')
        if viewed in labels:
            render_saved_analysis(store, user_id, labels[viewed])
    else:
        st.info('Run a single or batch prediction to start your saved analysis history.')


# Backwards compatibility alias for test suite
def render_workspace(store, user, current, validator=None):
    """Backwards compatible alias for test suite."""
    render_datasets_page(store, user, current)

