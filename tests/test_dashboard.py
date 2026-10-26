"""Exercise sign-in, protected pages and all four transaction model flows.

Every test uses an isolated SQLite database; this suite never creates or reads
accounts in the application's persistent workspace database.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import config
from src.workspace_store import WorkspaceStore


class TestDashboard(unittest.TestCase):
    EMAIL = 'reviewer@example.com'
    PASSWORD = 'FraudGuard-Test-2026!'
    PAGES = ['Dashboard', 'Datasets', 'Analyze Transactions', 'Review & History', 'Model Evaluation', 'About']
    PAGE_ALIASES = {
        'Overview': 'Dashboard',
        'My Workspace': 'Datasets',
        'Data Analysis': 'Datasets',
        'Fraud Prediction & Alerts': 'Analyze Transactions',
        'Model Performance': 'Model Evaluation',
        'Deep Learning': 'Model Evaluation',
        'About & Architecture': 'About'
    }

    def setUp(self):
        self.workspace_dir = tempfile.TemporaryDirectory(prefix='fraudguard-dashboard-')
        self.addCleanup(self.workspace_dir.cleanup)
        self.database_path = Path(self.workspace_dir.name) / 'workspace.sqlite3'
        self.database_patch = patch.object(config, 'WORKSPACE_DB_PATH',
                                           str(self.database_path), create=True)
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)

    def new_app(self):
        app = AppTest.from_file(str(BASE_DIR / 'dashboard' / 'app.py'), default_timeout=40).run()
        self.assertFalse(app.exception, str(app.exception))
        return app

    def widget(self, collection, label):
        matches = [item for item in collection if item.label == label]
        self.assertEqual(len(matches), 1, f'Expected one widget labelled {label!r}')
        return matches[0]

    def create_account(self, app):
        self.widget(app.text_input, 'Full name').set_value('Project Reviewer')
        self.widget(app.text_input, 'Email address').set_value(self.EMAIL)
        self.widget(app.text_input, 'Create password').set_value(self.PASSWORD)
        self.widget(app.text_input, 'Confirm password').set_value(self.PASSWORD)
        btn = [b for b in app.button if b.label in ('Create account', 'Create analyst account')][0]
        btn.click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertTrue(app.sidebar.radio, 'Creating an account should sign the user in')
        return app

    def sign_in(self, app, password=None):
        self.widget(app.text_input, 'Email').set_value(self.EMAIL)
        self.widget(app.text_input, 'Password').set_value(password or self.PASSWORD)
        self.widget(app.button, 'Sign in').click().run()
        self.assertFalse(app.exception, str(app.exception))
        return app

    def navigate(self, app, page):
        target = self.PAGE_ALIASES.get(page, page)
        self.widget(app.sidebar.radio, 'Navigation').set_value(target).run()
        self.assertFalse(app.exception, str(app.exception))

    def test_logged_out_data_is_protected(self):
        app = self.new_app()
        self.assertFalse(app.sidebar.radio, 'Navigation must require an account')
        self.assertFalse(app.dataframe, 'Transaction tables must not appear before sign-in')
        self.assertFalse(app.metric, 'Private workspace metrics must not appear before sign-in')
        self.widget(app.button, 'Sign in')
        self.assertTrue(any(b.label in ('Create account', 'Create analyst account') for b in app.button))

    def test_account_can_reopen_all_pages_after_sign_out(self):
        app = self.create_account(self.new_app())
        for page in self.PAGES:
            with self.subTest(page=page):
                self.navigate(app, page)

        self.widget(app.sidebar.button, 'Sign out').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.sidebar.radio)
        self.assertFalse(app.dataframe)

        # A fresh session verifies the account is persisted in SQLite rather
        # than surviving only in the previous browser's session state.
        reopened = self.sign_in(self.new_app())
        self.assertTrue(reopened.sidebar.radio)
        self.navigate(reopened, 'Datasets')


    def test_incorrect_password_does_not_open_workspace(self):
        app = self.create_account(self.new_app())
        self.widget(app.sidebar.button, 'Sign out').click().run()
        self.sign_in(app, password='Incorrect-password-2026!')
        self.assertFalse(app.sidebar.radio)
        self.assertTrue(app.error, 'Invalid credentials should provide an error')

    def test_all_models_can_score_a_transaction(self):
        app = self.create_account(self.new_app())
        self.navigate(app, 'Fraud Prediction & Alerts')
        for model in config.MODEL_FILES:
            with self.subTest(model=model):
                self.widget(app.selectbox, 'Active model').set_value(model).run()
                self.widget(app.button, 'Evaluate transaction').click().run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertEqual(app.session_state['single_result'][1]['Model_Used'], model)

        self.widget(app.sidebar.button, 'Sign out').click().run()
        reopened = self.sign_in(self.new_app())
        self.navigate(reopened, 'My Workspace')
        analyses = self.widget(reopened.selectbox, 'Saved analyses')
        self.assertEqual(len(analyses.options), len(config.MODEL_FILES),
                         'Each evaluated transaction should be saved to the account')
        analyses.select_index(0).run()
        self.widget(reopened.button, 'Open saved analysis').click().run()
        self.assertFalse(reopened.exception, str(reopened.exception))
        self.assertTrue(any('Model_Used' in table.value.columns for table in reopened.dataframe),
                        'Saved prediction results should reopen after a new sign-in')

    def test_default_single_comparison_evaluates_and_saves_all_models(self):
        app = self.create_account(self.new_app())
        self.navigate(app, 'Fraud Prediction & Alerts')
        self.assertEqual(self.widget(app.selectbox, 'Active model').value, 'Compare all models')
        self.widget(app.selectbox, 'Example transaction').set_value('Fraud example').run()
        self.widget(app.button, 'Evaluate transaction').click().run()
        self.assertFalse(app.exception, str(app.exception))
        report = app.session_state['comparison_single_result'][1]
        self.assertEqual(report['Models_Checked'], len(config.MODEL_FILES))
        self.assertEqual({row['Model'] for row in report['models']}, set(config.MODEL_FILES))
        self.assertEqual(len(report['models']), len(config.MODEL_FILES))
        self.assertEqual(report['Fraud_Votes'], len(report['Flagged_By']))
        self.assertGreater(report['Fraud_Votes'], 0,
                           'The fraud demonstration should expose at least one model flag')
        self.assertTrue(report['Transaction_ID'])
        for row in report['models']:
            with self.subTest(model=row['Model']):
                self.assertIn(row['Family'], ['Machine Learning', 'Deep Learning'])
                self.assertIn(row['Prediction'], ['LEGITIMATE', 'FRAUDULENT'])
                self.assertGreaterEqual(row['Score'], 0)
                self.assertLessEqual(row['Score'], 1)
                self.assertGreaterEqual(row['Threshold'], 0)
                self.assertLessEqual(row['Threshold'], 1)
        self.assertTrue(any('Why was this transaction flagged?' in item.value
                            for item in app.subheader))

        self.widget(app.sidebar.button, 'Sign out').click().run()
        reopened = self.sign_in(self.new_app())
        self.navigate(reopened, 'My Workspace')
        analyses = self.widget(reopened.selectbox, 'Saved analyses')
        self.assertEqual(len(analyses.options), 1)
        self.assertIn('All models', analyses.options[0])
        self.widget(reopened.button, 'Open saved analysis').click().run()
        self.assertFalse(reopened.exception, str(reopened.exception))
        self.assertTrue(reopened.dataframe)
        with WorkspaceStore(self.database_path) as store:
            user = store.authenticate(self.EMAIL, self.PASSWORD)
            runs = store.list_runs(user['id'])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]['model_name'], 'All models')
        self.assertEqual(runs[0]['kind'], 'single')
        self.assertEqual(runs[0]['summary']['rows'], 1)

    def test_switching_models_keeps_the_correct_saved_analysis_reference(self):
        app = self.create_account(self.new_app())
        self.navigate(app, 'Fraud Prediction & Alerts')
        self.widget(app.button, 'Evaluate transaction').click().run()
        comparison_id = app.session_state['comparison_single_result'][3]

        self.widget(app.selectbox, 'Active model').set_value('Random Forest').run()
        self.widget(app.button, 'Evaluate transaction').click().run()
        self.assertFalse(app.exception, str(app.exception))
        single_id = app.session_state['single_result'][2]
        self.assertNotEqual(single_id, comparison_id)
        self.assertTrue(any(f'Saved as analysis #{single_id}.' in item.value
                            for item in app.caption))

        self.widget(app.selectbox, 'Active model').set_value('Compare all models').run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertTrue(any(f'Saved as analysis #{comparison_id}.' in item.value
                            for item in app.caption))
        self.assertFalse(any(f'Saved as analysis #{single_id}.' in item.value
                             for item in app.caption))

    def test_saved_dataset_and_analysis_reopen_only_for_their_owner(self):
        sample = pd.read_csv(config.SAMPLE_TEST_PATH).head(2)
        csv_bytes = sample.to_csv(index=False).encode('utf-8')
        results = sample.assign(Prediction=['LEGITIMATE', 'FRAUDULENT'],
                                Model_Used='Random Forest')
        with WorkspaceStore(self.database_path) as store:
            user = store.create_user('Project Reviewer', self.EMAIL, self.PASSWORD)
            dataset = store.save_dataset(user['id'], 'Reviewer transactions', csv_bytes)
            run = store.save_run(user['id'], dataset['id'], 'Random Forest',
                                 results.to_csv(index=False).encode('utf-8'),
                                 {'rows': 2, 'flagged': 1, 'threshold': 0.5,
                                  'score_kind': 'fraud score'}, kind='batch')
            other = store.create_user('Another Reviewer', 'other@example.com', self.PASSWORD)
            other_dataset = store.save_dataset(other['id'], 'Private other data', csv_bytes)
            store.save_run(other['id'], other_dataset['id'], 'XGBoost',
                           results.to_csv(index=False).encode('utf-8'),
                           {'rows': 2, 'flagged': 1, 'threshold': 0.5,
                            'score_kind': 'fraud score'}, kind='batch')

        app = self.sign_in(self.new_app())
        self.navigate(app, 'My Workspace')
        datasets = self.widget(app.selectbox, 'Saved datasets')
        self.assertEqual(len(datasets.options), 1)
        self.assertIn('Reviewer transactions', datasets.options[0])
        datasets.select_index(0).run()
        self.widget(app.button, 'Open dataset').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertEqual(app.session_state['active_dataset_id'], dataset['id'])

        analyses = self.widget(app.selectbox, 'Saved analyses')
        self.assertEqual(len(analyses.options), 1)
        self.assertIn('Random Forest', analyses.options[0])
        analyses.select_index(0).run()
        self.widget(app.button, 'Open saved analysis').click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertEqual(app.session_state['saved_run_id'], run['id'])
        self.assertTrue(any('Prediction' in table.value.columns for table in app.dataframe))

        self.widget(app.sidebar.button, 'Sign out').click().run()
        reopened = self.sign_in(self.new_app())
        self.assertEqual(reopened.session_state['active_dataset_id'], dataset['id'],
                         'The last opened dataset should resume in a fresh session')

    def test_demo_batch_scan_survives_a_new_sign_in(self):
        app = self.create_account(self.new_app())
        self.navigate(app, 'My Workspace')
        self.widget(app.button, 'Add demo dataset').click().run()
        self.assertFalse(app.exception, str(app.exception))
        dataset_id = app.session_state['active_dataset_id']
        self.assertEqual(len(self.widget(app.selectbox, 'Saved datasets').options), 1)

        self.navigate(app, 'Fraud Prediction & Alerts')
        self.assertEqual(self.widget(app.selectbox, 'Active model').value, 'Compare all models')
        self.widget(app.radio, 'Prediction mode').set_value('Batch CSV').run()
        self.widget(app.radio, 'Batch data source').set_value('Saved workspace dataset').run()
        self.widget(app.button, 'Run batch scan').click().run()
        self.assertFalse(app.exception, str(app.exception))
        original_results = app.session_state['batch_result'][1].copy()
        self.assertEqual(len(original_results), len(pd.read_csv(config.SAMPLE_TEST_PATH)))
        self.assertTrue((original_results['Models_Checked'] == len(config.MODEL_FILES)).all())
        self.assertTrue(original_results['Fraud_Votes'].between(0, len(config.MODEL_FILES)).all())
        flagged = original_results.loc[original_results['Fraud_Votes'] > 0]
        self.assertGreater(len(flagged), 0)
        self.widget(app.checkbox, 'Show flagged transactions only').check().run()
        visible_results = [table.value for table in app.dataframe
                           if {'Source_Row', 'Fraud_Votes'}.issubset(table.value.columns)]
        self.assertEqual(len(visible_results), 1)
        self.assertEqual(len(visible_results[0]), len(flagged))
        self.assertTrue((visible_results[0]['Fraud_Votes'] > 0).all())
        self.widget(app.selectbox, 'Transaction to inspect').set_value(int(flagged.iloc[0]['Source_Row'])).run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertTrue(any('Why was this transaction flagged?' in item.value
                            for item in app.subheader))
        unflagged = original_results.loc[original_results['Fraud_Votes'] == 0]
        self.assertGreater(len(unflagged), 0)
        self.widget(app.checkbox, 'Show flagged transactions only').uncheck().run()
        inspector = self.widget(app.selectbox, 'Transaction to inspect')
        self.assertEqual(len(inspector.options), len(original_results),
                         'Clearing the flag filter must make unflagged rows inspectable')
        inspector.set_value(int(unflagged.iloc[0]['Source_Row'])).run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertTrue(any('Why was this transaction not flagged?' in item.value
                            for item in app.subheader))

        self.widget(app.sidebar.button, 'Sign out').click().run()
        reopened = self.sign_in(self.new_app())
        self.navigate(reopened, 'My Workspace')
        self.assertEqual(reopened.session_state['active_dataset_id'], dataset_id)
        self.assertEqual(len(self.widget(reopened.selectbox, 'Saved datasets').options), 1)
        analyses = self.widget(reopened.selectbox, 'Saved analyses')
        self.assertEqual(len(analyses.options), 1)
        self.assertIn('All models', analyses.options[0])
        self.widget(reopened.button, 'Open saved analysis').click().run()
        self.assertFalse(reopened.exception, str(reopened.exception))
        scored_tables = [table.value for table in reopened.dataframe
                         if set(original_results.columns).issubset(table.value.columns)]
        self.assertEqual(len(scored_tables), 1)
        pd.testing.assert_frame_equal(scored_tables[0], original_results,
                                      check_dtype=False, check_exact=False)


if __name__ == '__main__':
    unittest.main()
