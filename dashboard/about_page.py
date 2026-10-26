"""About, architecture and experiment context for the FraudGuard dashboard."""

import inspect
import json

import streamlit as st


def _count(value):
    return f"{int(value):,}" if value is not None else "Not recorded"


def _label(value):
    """Quote dynamic labels safely for the Graphviz source."""
    return json.dumps(str(value), ensure_ascii=False)


def _show_diagram(source):
    # Current Streamlit uses width; older supported releases use the boolean.
    if 'width' in inspect.signature(st.graphviz_chart).parameters:
        st.graphviz_chart(source, width='stretch')
    else:
        st.graphviz_chart(source, use_container_width=True)


def _training_architecture(metadata):
    splits = metadata.get('splits', {})
    split_labels = []
    for key, title in [('train', 'Train'), ('validation', 'Validation'), ('test', 'Test')]:
        rows = splits.get(key, {}).get('rows')
        split_labels.append(f'{title}: {_count(rows)} rows')
    partition_label = '\n'.join(['Stratified partitions'] + split_labels)
    feature_count = len(metadata.get('feature_names', []))
    preprocess_label = f'Training Preprocessing\nImpute, engineer {feature_count} features, scale'
    best_model = metadata.get('best_model', 'Validation-selected model')
    selection_label = f'Validation Only\nF1 thresholds & model selection\nSelected: {best_model}'
    shap_model = metadata.get('shap_model', 'selected tree model')
    return f'''digraph training {{
        graph [rankdir=TB, bgcolor="transparent", pad="0.1", nodesep="0.2", ranksep="0.28", fontname="Arial"];
        node [shape=box, style="rounded,filled", fillcolor="#eff6ff", color="#93c5fd", fontcolor="#0f2746", fontname="Arial", fontsize=10, margin="0.12,0.08"];
        edge [color="#64748b", arrowsize=0.6, fontname="Arial", fontsize=9];
        source [label="Synthetic benchmark\n15,000 transactions; schema checks"];
        split [label={_label(partition_label)}];
        prep [label={_label(preprocess_label)}];
        balanced [label="Branch A: SMOTE-NC\nBalanced training set", fillcolor="#ecfdf5", color="#6ee7b7"];
        normal [label="Branch B: Normal Only\nLegitimate training rows", fillcolor="#f5f3ff", color="#c4b5fd"];
        trees [label="ML: Random Forest + XGBoost", fillcolor="#ecfdf5", color="#6ee7b7"];
        classifier [label="DL: Supervised Neural Network", fillcolor="#ecfdf5", color="#6ee7b7"];
        autoencoder [label="DL: Reconstruction Autoencoder", fillcolor="#f5f3ff", color="#c4b5fd"];
        selection [label={_label(selection_label)}, fillcolor="#fff7ed", color="#fdba74"];
        evaluation [label="Final Test Partition\nPrecision, Recall, F1, Average Precision"];
        artifacts [label={_label(f'Saved Artifacts\nThresholds, metadata, charts, Tree SHAP: {shap_model}')}, fillcolor="#e0f2fe", color="#7dd3fc"];
        source -> split -> prep;
        prep -> balanced;
        prep -> normal;
        balanced -> trees;
        balanced -> classifier;
        normal -> autoencoder;
        trees -> selection;
        classifier -> selection;
        autoencoder -> selection;
        selection -> evaluation -> artifacts;
        {{rank=same; balanced; normal;}}
        {{rank=same; trees; classifier; autoencoder;}}
    }}'''


def _workspace_architecture(model_count):
    return f'''digraph workspace {{
        graph [rankdir=LR, bgcolor="transparent", pad="0.1", nodesep="0.2", ranksep="0.28", fontname="Arial"];
        node [shape=box, style="rounded,filled", fillcolor="#eff6ff", color="#93c5fd", fontcolor="#0f2746", fontname="Arial", fontsize=10, margin="0.1,0.08"];
        edge [color="#64748b", arrowsize=0.6, fontname="Arial", fontsize=9];
        account [label="1. Sign In\nLocal analyst session"];
        data [label="2. Datasets\nUpload & validate CSV"];
        score [label={_label(f'3. Screen\nEvaluate with {model_count} models')}];
        history [label="4. Review\nStatus, notes & audit", fillcolor="#ecfdf5", color="#6ee7b7"];
        resume [label="5. History\nExport results & audit", fillcolor="#ecfdf5", color="#6ee7b7"];
        account -> data -> score -> history -> resume;
    }}'''


def _architecture_tab(metadata, model_count):
    st.subheader('Application Architecture & Workflow')
    st.info('🛡️ **System Scope: Post-Transaction Fraud Screening**\n\n'
            'Because closing-balance fields (`newbalanceOrig`, `newbalanceDest`) are only recorded after transaction completion, '
            'this system operates as a **post-transaction fraud screening and investigative tool** rather than a pre-authorization payment gateway. '
            'Predictions assist human fraud analysts in review; the system does not execute banking transactions or automatically block customer bank accounts.')
    
    st.markdown('#### A. Offline Training & Validation Pipeline')
    _show_diagram(_training_architecture(metadata))
    st.caption('Imputation, scaling, and SMOTE-NC oversampling are fitted on training data only. Validation selects models and thresholds; test evaluation is unbiased and frozen.')

    st.markdown('#### B. Analyst Screening & Review Workflow')
    _show_diagram(_workspace_architecture(model_count))
    st.caption('Uploaded transaction datasets run on frozen pre-trained models. Uploads do not retrain models.')

    st.info(
        'Uploading a CSV adds it to your workspace. Scoring applies the existing trained model; '
        'it does not retrain a model or add your uploaded rows to the benchmark.'
    )
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('**What you can return to**')
            st.write(
                'Saved datasets and prediction runs belong to the signed-in account. '
                'Open your history to review earlier results or download a scored file. '
                'Signing out ends the active session and keeps saved work for your next sign-in '
                'to this project on the same computer.'
            )
    with right:
        with st.container(border=True):
            st.markdown('**What a prediction means**')
            st.write(
                'Each row receives a model score, a threshold-based fraud verdict and an advisory '
                'risk band. Results support manual investigation. This project does not execute '
                'payments, block accounts or verify whether a transaction is truly fraudulent.'
            )


def _neural_details(model, feature_count, autoencoder=False):
    layers = model.get('hidden_layers', [])
    layer_sizes = [feature_count] + layers + [feature_count if autoencoder else 1]
    st.markdown('**' + ' → '.join(str(size) for size in layer_sizes) + '**')
    if autoencoder:
        st.write(
            'Learns to reconstruct legitimate training transactions with mean squared error. '
            'A higher reconstruction error signals unusual activity. Its anomaly score is '
            'normalized for display and is not a calibrated probability of fraud.'
        )
        st.caption(
            f"Original legitimate training examples: {_count(model.get('legitimate_training_rows'))}. "
            'Early stopping monitors legitimate validation reconstruction error; labelled validation '
            'examples choose the final anomaly threshold.'
        )
    else:
        st.write(
            'A supervised feed-forward network learns fraud labels with ReLU hidden layers, '
            'a sigmoid output, binary cross-entropy and Adam. Training uses SMOTE-NC-balanced '
            'training data, with L2 regularization and validation-based early stopping.'
        )
    st.caption(
        f"Parameters: {_count(model.get('parameters'))} · "
        f"Epochs run: {_count(model.get('epochs_run'))} · "
        f"Selected epoch: {_count(model.get('best_epoch'))}"
    )


def _methods_tab(metadata):
    models = metadata.get('models', {})
    feature_count = len(metadata.get('feature_names', []))
    left, right = st.columns(2)
    for column, name, description in [
        (left, 'Random Forest', 'An ensemble of decision trees combines multiple views of the training data to classify fraud.'),
        (right, 'XGBoost', 'Gradient-boosted trees progressively correct earlier errors to learn transaction patterns.'),
    ]:
        with column:
            with st.container(border=True):
                st.caption('MACHINE LEARNING')
                st.subheader(name)
                st.write(description)
                threshold = models.get(name, {}).get('threshold')
                if threshold is not None:
                    st.caption(f'Validation-selected decision threshold: {threshold:.6f}')
    for column, name, autoencoder in [
        (left, 'Deep Neural Network', False), (right, 'Deep Autoencoder', True)
    ]:
        with column:
            with st.container(border=True):
                st.caption('DEEP LEARNING')
                st.subheader(name)
                _neural_details(models.get(name, {}), feature_count, autoencoder)
    st.caption(
        'Both neural models use the CPU backpropagation implementation in scikit-learn. '
        'Their hidden-layer sizes, parameter counts and training epochs above come from the saved experiment metadata.'
    )
    st.markdown('#### Explanations and score interpretation')
    st.write(
        f"Tree SHAP explains **{metadata.get('shap_model', 'the selected tree model')}** in the "
        'Model Performance page. Those feature attributions apply to that tree model only; '
        'they are not explanations of the neural network or the autoencoder. Individual '
        'transaction rule indicators are separate heuristics.'
    )
    st.write(
        'All displayed scores are uncalibrated. LOW, MEDIUM, HIGH and CRITICAL are illustrative '
        'score bands at 30%, 60% and 80%; the fraud verdict uses the model’s separately selected '
        'validation threshold. A risk band and verdict can therefore differ.'
    )


def _evaluation_tab(metadata):
    dataset = metadata.get('dataset', {})
    st.subheader('Read the results in context')
    st.write(dataset.get('provenance', 'Locally generated synthetic benchmark; not real bank transactions'))
    rows = dataset.get('rows', 0)
    fraud_count = dataset.get('fraud_count', 0)
    st.write(
        f'The recorded benchmark contains **{_count(rows)} transactions** and '
        f'**{_count(fraud_count)} labelled fraud examples**'
        + (f' ({fraud_count / rows:.2%} fraud prevalence).' if rows else '.')
    )
    splits = metadata.get('splits', {})
    split_total = sum(part.get('rows', 0) for part in splits.values())
    partition_rows = []
    purposes = {
        'train': 'Fit preprocessing and model weights; oversampling only here',
        'validation': 'Choose checkpoints, decision thresholds and the default model',
        'test': 'Report final performance after all selection decisions',
    }
    for name in ['train', 'validation', 'test']:
        part = splits.get(name, {})
        count = part.get('rows', 0)
        partition_rows.append({
            'Partition': name.title(),
            'Rows': count,
            'Share': f'{count / split_total:.0%}' if split_total else 'Not recorded',
            'Fraud examples': part.get('fraud_count', 0),
            'Purpose': purposes[name],
        })
    st.dataframe(partition_rows, hide_index=True, width='stretch')
    st.markdown(f"**Validation-selected benchmark model:** {metadata.get('best_model', 'Not recorded')}")
    st.caption('Fraud Prediction & Alerts compares all four models by default; each uses its own validation-selected decision threshold.')
    st.write('Selection rule: ' + metadata.get('selection_rule', 'Not recorded') + '.')
    st.write('Threshold rule: ' + metadata.get('threshold_rule', 'Not recorded') + '.')
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('**Metrics with a purpose**')
            st.write(
                'Precision measures how many flagged examples are fraud. Recall measures how '
                'much labelled fraud is detected. F1 balances the two. Average precision '
                '(shown as PR-AUC) summarizes the precision-recall tradeoff; ROC-AUC measures '
                'ranking across thresholds. Accuracy alone can hide rare-class failures.'
            )
    with right:
        with st.container(border=True):
            st.markdown('**Limits that affect deployment**')
            st.write(
                'The data is locally generated, and evaluation uses random stratified '
                'partitions rather than an independent real-bank or later-time dataset. '
                'Closing balances are required, so the current design is a retrospective '
                'transaction review tool; those fields may be unavailable before a payment completes.'
            )
    st.caption(
        f"Recorded random seed: {metadata.get('random_state', 'Not recorded')}. "
        f"Experiment created (UTC): {metadata.get('created_utc', 'Not recorded')}. "
        'The measured benchmark results do not establish performance on real customers.'
    )
    with st.expander('Recorded dataset fingerprint and feature schema'):
        st.write('Dataset SHA-256: ' + dataset.get('sha256', 'Not recorded'))
        st.write('Engineered features: ' + ', '.join(metadata.get('feature_names', [])))


def _components_tab():
    st.subheader('What each part owns')
    st.dataframe([
        {'Component': 'Streamlit interface', 'Responsibility': 'Sign-in, dataset preview, single/batch scoring, charts and downloads'},
        {'Component': 'Account and workspace storage', 'Responsibility': 'Local accounts and per-user datasets and saved prediction runs in SQLite'},
        {'Component': 'Data preprocessor', 'Responsibility': 'Validate columns and values; apply stored imputation, engineered features and scaling'},
        {'Component': 'Fraud predictor', 'Responsibility': 'Load a chosen saved model and its threshold; produce row-level scores and advisory outputs'},
        {'Component': 'Model comparison and transaction inspection', 'Responsibility': 'Score with all four models, show individual flags and thresholds, and separate quantitative observations from supplied fraud labels'},
        {'Component': 'Training pipeline', 'Responsibility': 'Create the partitions; train ML/DL models; select on validation; evaluate on test'},
        {'Component': 'Experiment artifacts', 'Responsibility': 'Saved model/preprocessor files, metadata, comparison metrics, curves and tree SHAP figures'},
    ], hide_index=True, width='stretch')
    st.markdown('#### Local account scope')
    st.write(
        'Accounts use salted PBKDF2-derived password hashes. Your datasets and runs are associated '
        'with your user account in a local SQLite database on this computer. Accounts and '
        'saved work are local to this project; cloud sync is not configured. The application does not encrypt '
        'that database at rest, and local account creation does not verify an email address. '
        'Keep this demonstration and its data on a trusted computer.'
    )
    st.markdown('#### Next research and engineering steps')
    st.write(
        'Validate on an independently sourced dataset and a later-time test split; calibrate '
        'scores and measure the cost of missed fraud versus false alerts. Account-linked '
        'transaction histories would enable sequence models such as LSTMs. Hosted deployment '
        'would also need managed identity, recovery, access auditing, encrypted storage and '
        'a defined retention policy. Streaming ingestion and live payment decisions remain future work.'
    )


def render_about_page(metadata):
    """Render an accurate product and architecture page from saved experiment metadata."""
    metadata = metadata or {}
    model_count = len(metadata.get('models', {}))
    feature_count = len(metadata.get('feature_names', []))
    st.header('About & architecture')
    st.write(
        '**FraudGuard AI brings machine learning, deep learning and a saved personal workspace '
        'into one financial-fraud investigation demonstration.** Compare trained models, '
        'review suspicious transactions and return to your own datasets and results after signing in.'
    )
    with st.container(border=True):
        columns = st.columns(4)
        columns[0].metric('Trained models', model_count)
        columns[1].metric('Engineered features', feature_count)
        columns[2].metric('Benchmark transactions', _count(metadata.get('dataset', {}).get('rows')))
        columns[3].metric('Validation-selected model', metadata.get('best_model', 'Not recorded'))
    tabs = st.tabs(['How it works', 'ML + DL methods', 'Data & evaluation', 'Components & scope'])
    with tabs[0]:
        _architecture_tab(metadata, model_count)
    with tabs[1]:
        _methods_tab(metadata)
    with tabs[2]:
        _evaluation_tab(metadata)
    with tabs[3]:
        _components_tab()
