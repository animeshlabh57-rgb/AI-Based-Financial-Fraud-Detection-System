"""Review & History: Review flagged transactions, manage analyst notes and review statuses, and export audit trails."""

import io
import pandas as pd
import streamlit as st
from dashboard.workspace_page import csv_frame
from dashboard.transaction_explanation import render_transaction_explanation

REVIEW_STATUSES = ['Pending', 'Reviewed', 'Escalated']


def render_review_page(store, user, metadata):
    """Review & History section."""
    user_id = user['id']
    st.header('Review & History')
    st.write('Inspect flagged transactions, document analyst conclusions, and track review status across your saved runs.')

    runs = store.list_runs(user_id)
    if not runs:
        st.info('No saved analyses found yet. Run an evaluation from **Analyze Transactions** to begin building your analysis history.')
        return

    labels = {r['id']: r for r in runs}
    run_choices = list(labels)

    chosen_run_id = st.selectbox(
        'Select saved analysis run',
        run_choices,
        format_func=lambda rid: f'#{rid} · {labels[rid]["model_name"]} ({labels[rid]["kind"].title()}) · {labels[rid]["created_at"][:19]} · {labels[rid]["summary"].get("rows", 0)} rows ({labels[rid]["summary"].get("flagged", 0)} flagged)'
    )

    selected_run = store.get_run(user_id, chosen_run_id)
    if not selected_run:
        st.warning('Selected analysis could not be loaded.')
        return

    summary = selected_run.get('summary', {})
    res_df = csv_frame(selected_run['result_csv'])

    # Summary metrics for the run
    st.subheader(f'Analysis #{chosen_run_id}: {selected_run["model_name"]}')
    st.caption(f'Execution type: **{selected_run["kind"].title()}** · Saved timestamp: **{selected_run["created_at"]}**')

    m1, m2, m3, m4 = st.columns(4)
    total_rows = summary.get('rows', len(res_df))
    flagged_rows = summary.get('flagged', 0)
    pct = (flagged_rows / total_rows * 100) if total_rows else 0.0

    m1.metric('Total Transactions', f'{total_rows:,}')
    m2.metric('Flagged for Review', f'{flagged_rows:,}')
    m3.metric('Flagged Share', f'{pct:.2f}%')
    m4.metric('Model Used', selected_run['model_name'])

    st.divider()

    # Identify flagged rows in the result dataframe
    if 'Fraud_Votes' in res_df.columns:
        flagged_mask = res_df['Fraud_Votes'] > 0
    elif 'Prediction' in res_df.columns:
        flagged_mask = res_df['Prediction'].isin(['FRAUDULENT', 'FLAGGED FOR REVIEW'])
    else:
        flagged_mask = pd.Series([False] * len(res_df))

    flagged_df = res_df.loc[flagged_mask]

    # Fetch existing reviews for this run
    existing_reviews = {r['source_row']: r for r in store.list_reviews(user_id, chosen_run_id)}

    # Attach current review status and notes to display
    display_df = flagged_df.copy()
    if not display_df.empty:
        display_df['Review_Status'] = [
            existing_reviews.get(int(row.get('Source_Row', idx + 1)), {}).get('status', 'Pending')
            for idx, row in display_df.iterrows()
        ]
        display_df['Reviewer_Notes'] = [
            existing_reviews.get(int(row.get('Source_Row', idx + 1)), {}).get('notes', '')
            for idx, row in display_df.iterrows()
        ]

    st.subheader('Flagged Transactions Queue')
    st.caption('Review each flagged transaction, document findings, and assign a review status (Pending, Reviewed, or Escalated).')

    if display_df.empty:
        st.success('✅ No transactions were flagged for review in this run.')
    else:
        status_filter = st.radio('Filter by review status', ['All Flagged', 'Pending', 'Reviewed', 'Escalated'], horizontal=True)
        if status_filter != 'All Flagged':
            view_queue = display_df[display_df['Review_Status'] == status_filter]
        else:
            view_queue = display_df

        priority_cols = ['Source_Row', 'Transaction_ID', 'Review_Status', 'type', 'amount', 'Reviewer_Notes']
        visible_cols = [c for c in priority_cols if c in view_queue.columns]
        visible_cols += [c for c in view_queue.columns if c not in visible_cols]

        st.dataframe(view_queue, hide_index=True, width='stretch', column_order=visible_cols)

        # Select transaction to review
        st.markdown('#### Analyst Review & Notes')
        
        row_options = display_df['Source_Row'].tolist() if 'Source_Row' in display_df.columns else list(range(1, len(display_df) + 1))
        selected_source_row = st.selectbox(
            'Select transaction row to review',
            row_options,
            format_func=lambda srow: f'Row {srow} · Current Status: {existing_reviews.get(int(srow), {}).get("status", "Pending")}'
        )

        curr_review = existing_reviews.get(int(selected_source_row), {})
        current_status = curr_review.get('status', 'Pending')
        current_notes = curr_review.get('notes', '')

        with st.form(f'review_form_{chosen_run_id}_{selected_source_row}'):
            st.markdown(f'**Reviewing Row #{selected_source_row}**')
            col_stat, col_user = st.columns([1, 1])
            with col_stat:
                new_status = st.selectbox('Review Decision Status', REVIEW_STATUSES,
                                          index=REVIEW_STATUSES.index(current_status))
            with col_user:
                st.text_input('Reviewing Analyst', value=user['display_name'], disabled=True)

            new_notes = st.text_area('Analyst Notes / Evidence Justification',
                                     value=current_notes,
                                     max_chars=5000,
                                     placeholder='Document findings: e.g. "Customer confirmed overseas travel; legitimate high-value withdrawal." or "Unrecognized recipient account; escalated to fraud ops."')

            save_submitted = st.form_submit_button('Save Review Decision', type='primary')

        if save_submitted:
            try:
                saved_rev = store.save_review(
                    user_id=user_id,
                    run_id=chosen_run_id,
                    source_row=int(selected_source_row),
                    status=new_status,
                    notes=new_notes.strip(),
                    reviewer_display_name=user['display_name']
                )
                st.success(f'Review for Row #{selected_source_row} saved successfully as **{new_status}**.')
                st.rerun()
            except Exception as e:
                st.error(f'Failed to save review: {e}')

        # Show audit trail
        audit_trail = store.list_review_audit(user_id, chosen_run_id, int(selected_source_row))
        if audit_trail:
            with st.expander(f'Audit History for Row #{selected_source_row} ({len(audit_trail)} events)'):
                audit_display = pd.DataFrame(audit_trail)[['status', 'notes', 'reviewer_display_name', 'updated_at']]
                audit_display.columns = ['Status', 'Analyst Notes', 'Reviewer', 'Timestamp (UTC)']
                st.dataframe(audit_display, hide_index=True, width='stretch')

        # Detailed Transaction Inspection
        st.markdown('#### Transaction Inspection')
        tx_row = res_df.loc[res_df['Source_Row'] == selected_source_row] if 'Source_Row' in res_df.columns else res_df.iloc[[selected_source_row - 1]]
        if not tx_row.empty:
            tx_dict = tx_row.iloc[0].to_dict()
            render_transaction_explanation(tx_dict, show_model_table=True)

    st.divider()

    # Download exports
    st.subheader('Export Analysis & Review Audit')
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            '📥 Download full run results (CSV)',
            selected_run['result_csv'],
            file_name=f'analysis_run_{chosen_run_id}.csv',
            mime='text/csv',
            width='stretch'
        )
    with exp_col2:
        # Generate review audit report CSV
        all_reviews = store.list_reviews(user_id, chosen_run_id)
        if all_reviews:
            audit_df = pd.DataFrame(all_reviews)[['run_id', 'source_row', 'status', 'notes', 'reviewer_display_name', 'updated_at']]
            audit_df.columns = ['Run_ID', 'Source_Row', 'Review_Status', 'Analyst_Notes', 'Reviewer', 'Updated_At_UTC']
            st.download_button(
                '📋 Download review decisions report (CSV)',
                audit_df.to_csv(index=False).encode('utf-8'),
                file_name=f'review_audit_run_{chosen_run_id}.csv',
                mime='text/csv',
                width='stretch'
            )
        else:
            st.caption('No reviews recorded yet for this run.')
