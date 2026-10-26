"""Row-level observations, kept separate from model predictions and labels."""

from collections.abc import Mapping
import math

import config


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _units(value):
    return f'{value:,.2f} units'


def _models(model_rows):
    decisions = []
    if not isinstance(model_rows, (list, tuple)):
        return decisions
    for row in model_rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get('Model') or 'Unnamed model')
        score, threshold = _number(row.get('Score')), _number(row.get('Threshold'))
        decision = {
            'Model': name,
            'Family': str(row.get('Family') or config.MODEL_FAMILIES.get(name, 'Not supplied')),
            'Prediction': 'Unavailable',
            'Score': score,
            'Threshold': threshold,
            'Score_Kind': str(row.get('Score_Kind') or 'Score type not supplied'),
        }
        if score is not None and threshold is not None:
            decision['Prediction'] = 'FRAUDULENT' if score >= threshold else 'LEGITIMATE'
        reported = row.get('Prediction')
        if reported in ('FRAUDULENT', 'LEGITIMATE') and reported != decision['Prediction']:
            decision['Reported_Prediction'] = reported
        reconstruction_error = _number(row.get('Reconstruction_Error'))
        if reconstruction_error is not None:
            decision['Reconstruction_Error'] = reconstruction_error
        decisions.append(decision)
    return decisions


def build_transaction_explanation(raw_tx, model_rows=None):
    """Describe a row without inferring ground truth or attributing model causality.

    ``model_rows`` accepts Model, Family, Prediction, Score, Threshold and
    Score_Kind; optional Reconstruction_Error is retained when supplied.
    Decisions are rechecked from unrounded Score >= Threshold. Missing raw
    numeric inputs are never replaced by zero for these observations.
    """
    raw_tx = raw_tx if isinstance(raw_tx, Mapping) else {}
    values = {}
    missing = []
    for field in config.RAW_FEATURE_COLUMNS:
        if field == 'type':
            continue
        value = _number(raw_tx.get(field))
        if value is None or value < 0:
            missing.append(field)
        else:
            values[field] = value
    supplied_type = raw_tx.get('type')
    transaction_type = supplied_type.strip().upper() if isinstance(supplied_type, str) else ''
    if transaction_type not in config.TRANSACTION_TYPES:
        missing.append('type')
        transaction_type = ''

    indicators = []
    measurements = dict(values)

    def add(indicator, observed, meaning):
        indicators.append({'Indicator': indicator, 'Observed': observed, 'Meaning': meaning})

    if transaction_type:
        high_review_type = transaction_type in ('TRANSFER', 'CASH_OUT')
        add('Transaction type', transaction_type,
            'A predefined review category in this demonstration. Type alone does not establish fraud.'
            if high_review_type else
            'Balance direction and recipient reporting depend on this transaction type. Type alone does not establish legitimacy.')

    amount = values.get('amount')
    old_origin, new_origin = values.get('oldbalanceOrg'), values.get('newbalanceOrig')
    old_dest, new_dest = values.get('oldbalanceDest'), values.get('newbalanceDest')
    if amount is not None:
        add('Transaction amount', _units(amount),
            'Exceeds the illustrative 100,000-unit review rule. A large legitimate transaction can also trigger this rule.'
            if amount > 100000 else
            'At or below the illustrative 100,000-unit review rule. This does not determine whether the transaction is legitimate.')

    if old_origin is not None and new_origin is not None:
        debit = old_origin - new_origin
        measurements['sender_debit'] = debit
        add('Recorded sender balance change',
            f'{_units(old_origin)} → {_units(new_origin)}; opening − closing = {_units(debit)}',
            'A positive value is a recorded debit; a negative value is a recorded credit. CASH_IN normally credits the sender account.')
        measurements['sender_drained'] = old_origin > 0 and new_origin == 0
        if old_origin > 0 and new_origin == 0:
            add('Sender balance depletion', 'Closing balance is zero after a positive opening balance',
                'The recorded balance was fully depleted. Legitimate withdrawals and transfers can also empty an account.')
        if amount is not None:
            gap = debit - amount
            measurements['sender_debit_gap'] = gap
            measurements['errorBalanceOrig'] = new_origin + amount - old_origin
            if transaction_type == 'CASH_IN':
                meaning = ('CASH_IN normally increases the sender balance, so a debit-minus-amount '
                           'gap is descriptive and is not a debit consistency test for this type.')
            elif transaction_type in ('TRANSFER', 'CASH_OUT', 'DEBIT', 'PAYMENT'):
                meaning = ('A single fully recorded outgoing posting would debit approximately the amount. '
                           'A gap can reflect reporting conventions, fees or timing and does not prove fraud.')
            else:
                meaning = 'The transaction direction is unknown, so the debit consistency assumption cannot be checked.'
            add('Sender debit versus transaction amount',
                f'(opening − closing) − amount = {_units(gap)}', meaning)
            add('Engineered sender balance feature',
                f'errorBalanceOrig = closing + amount − opening = {_units(measurements["errorBalanceOrig"])}',
                'This is the same sender mismatch with the opposite sign, not an independent second piece of evidence.')

    if amount is not None and old_origin is not None:
        if old_origin > 0:
            ratio = amount / old_origin
            measurements['amount_to_opening_balance'] = ratio
            add('Amount / sender opening balance', f'{ratio:.2%}',
                'This is an incoming amount for CASH_IN; the ratio does not describe funds leaving the sender.'
                if transaction_type == 'CASH_IN' else
                ('Exceeds the illustrative 85% review rule. The ratio alone does not establish fraud.'
                 if ratio > 0.85 else 'Does not exceed the illustrative 85% review rule. A lower ratio does not establish legitimacy.'))
        else:
            add('Amount / sender opening balance', 'Undefined: opening balance is zero',
                'No percentage is calculated because division by zero is undefined. The trained feature pipeline uses its own stored convention.')

    if old_dest is not None and new_dest is not None:
        credit = new_dest - old_dest
        measurements['recipient_credit'] = credit
        add('Recorded recipient balance change',
            f'{_units(old_dest)} → {_units(new_dest)}; closing − opening = {_units(credit)}',
            'This is the change in the supplied recipient snapshot; some transaction types do not record a complete recipient or merchant balance.')
        if amount is not None:
            dest_gap = amount - credit
            measurements['recipient_credit_gap'] = dest_gap
            measurements['errorBalanceDest'] = old_dest + amount - new_dest
            if transaction_type == 'TRANSFER':
                meaning = ('A simplified fully recorded transfer would credit the recipient by approximately '
                           'the amount. A discrepancy can warrant review but is not proof of fraud.')
            elif transaction_type == 'CASH_OUT':
                meaning = ('Cash-out recipient or outlet balances may be absent or incomplete. '
                           'Do not interpret an unrecorded recipient credit as proof of fraud.')
            elif transaction_type in ('DEBIT', 'PAYMENT'):
                meaning = ('Merchant or recipient balances may not be tracked for this type. '
                           'The gap alone is not a reliable fraud test.')
            elif transaction_type == 'CASH_IN':
                meaning = ('Cash-in follows a different funds direction. This recipient-credit formula '
                           'is an engineered observation, not a balanced-transfer test for this type.')
            else:
                meaning = 'Recipient reporting and transaction direction are unknown; the gap cannot establish fraud.'
            add('Recipient credit gap / engineered feature',
                f'amount − (closing − opening) = errorBalanceDest = {_units(dest_gap)}', meaning)
    if old_dest == 0 and transaction_type in ('TRANSFER', 'CASH_OUT'):
        measurements['recipient_opening_zero'] = True
        add('Zero recorded recipient opening balance', _units(old_dest),
            'Useful context for reviewing a transfer or cash-out. A zero snapshot does not prove this is a newly opened account or a fraudulent destination.')

    label_value = _number(raw_tx.get(config.TARGET_COLUMN))
    label = None
    if label_value in (0, 1):
        label_name = 'FRAUDULENT' if label_value == 1 else 'LEGITIMATE'
        label = f'Provided dataset label: {label_name} (isFraud={int(label_value)}). This supplied label is separate from every model prediction.'

    decisions = _models(model_rows)
    known = [row for row in decisions if row['Prediction'] != 'Unavailable']
    if known:
        decisions_text = '; '.join(
            f'{row["Model"]}: score {row["Score"]:.6f} '
            f'{"≥" if row["Prediction"] == "FRAUDULENT" else "<"} '
            f'threshold {row["Threshold"]:.6f} → {row["Prediction"]}'
            for row in known)
        summary = 'Threshold decisions — ' + decisions_text + '. These are model predictions, not verified fraud findings.'
    else:
        summary = ('No complete model score and decision threshold were supplied. '
                   'A fraud verdict cannot be determined from these observations alone.')
    if any('Reported_Prediction' in row for row in decisions):
        summary += ' A supplied prediction could not be reproduced from the displayed score and threshold; verify the saved model details and rounding.'
    return {
        'summary': summary,
        'indicators': indicators,
        'label': label,
        'model_decisions': decisions,
        'measurements': measurements,
        'missing_inputs': missing,
        'explanation_note': ('The quantitative observations below are predefined checks, not model feature '
                             'attributions or causal explanations. Missing observations do not imply legitimacy. '
                             'Scores are not calibrated probabilities of fraud.'),
    }


def render_transaction_explanation(tx, model_rows=None, selected_model=None, *, show_model_table=True):
    """Render model decisions, optional supplied ground truth and quantitative checks."""
    import streamlit as st

    explanation = build_transaction_explanation(tx, model_rows)
    decisions = explanation['model_decisions']
    if any(row['Prediction'] == 'FRAUDULENT' for row in decisions):
        st.subheader('Why was this transaction flagged?')
    elif any(row['Prediction'] == 'LEGITIMATE' for row in decisions):
        st.subheader('Why was this transaction not flagged?')
    else:
        st.subheader('Understand this transaction')
    st.write(explanation['summary'])
    if explanation['label']:
        st.info(explanation['label'])
    else:
        st.caption('No valid isFraud label was provided. The actual fraud status is unknown.')
    if decisions and show_model_table:
        st.dataframe(decisions, hide_index=True, width='stretch', column_config={
            'Score': st.column_config.NumberColumn(format='%.6f'),
            'Threshold': st.column_config.NumberColumn(format='%.6f'),
        })
    selected = next((row for row in decisions if row['Model'] == selected_model), None)
    if selected:
        if selected['Model'] == 'Deep Autoencoder' or 'reconstruction' in selected['Score_Kind'].lower():
            st.write(
                'The autoencoder learned to reconstruct legitimate training examples. Its anomaly '
                'score summarizes reconstruction error; unusual transactions can also be legitimate. '
                'The observations below do not identify which features caused its anomaly score.'
            )
            if 'Reconstruction_Error' in selected:
                st.caption(f'Provided raw reconstruction error: {selected["Reconstruction_Error"]:.6g}.')
            else:
                st.caption('Raw reconstruction error was not supplied; the available model score is shown above.')
        elif selected['Family'] == 'Deep Learning':
            st.write(
                'The neural classifier uses learned nonlinear combinations of the input features. '
                'These balance observations help you inspect the row; they are not attributions '
                'of the neural network’s individual decision.'
            )
        else:
            st.caption(
                'These row observations are separate from SHAP. The global tree SHAP charts on '
                'Model Performance describe the recorded tree model and do not explain this individual transaction.'
            )
    st.caption(explanation['explanation_note'])
    if explanation['indicators']:
        st.dataframe(explanation['indicators'], hide_index=True, width='stretch')
    else:
        st.info('No quantitative observations can be calculated from the supplied fields.')
    if explanation['missing_inputs']:
        st.caption('Missing or unusable observation inputs: ' + ', '.join(explanation['missing_inputs']) + '.')
    return explanation
