"""Security, isolation, and restart persistence for local account workspaces."""

import hashlib
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workspace_store import MAX_CSV_BYTES, PASSWORD_ITERATIONS, WorkspaceStore


class TestWorkspaceStore(unittest.TestCase):
    password = 'project-demo-password-2026'
    csv_bytes = b'amount,type\n100,TRANSFER\n50,CASH_OUT\n'
    result_csv = b'amount,prediction\n100,Fraud\n50,Legitimate\n'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / 'nested' / 'workspaces.sqlite3'
        self.store = WorkspaceStore(self.db_path)
        self.alice = self.store.create_user('\u0905\u0928\u0928\u094d\u092f\u093e Alice', '  ALICE@Example.COM  ', self.password)
        self.bob = self.store.create_user('Bob', 'bob@example.com', self.password)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def dataset(self, user=None, name='transactions.csv'):
        return self.store.save_dataset((user or self.alice)['id'], name, self.csv_bytes)

    def run_record(self, dataset_id=None, user=None, kind='batch'):
        return self.store.save_run((user or self.alice)['id'], dataset_id, 'Random Forest',
                                   self.result_csv, {'total': 2, 'fraud': 1}, kind=kind)

    def test_unicode_user_and_normalized_authentication(self):
        self.assertEqual(self.alice['email'], 'alice@example.com')
        self.assertIn('\u0905\u0928\u0928\u094d\u092f\u093e', self.alice['display_name'])
        self.assertEqual(self.store.authenticate(' Alice@EXAMPLE.com ', self.password), self.alice)
        self.assertIsNone(self.store.authenticate('alice@example.com', 'incorrect password'))
        self.assertIsNone(self.store.authenticate('missing@example.com', self.password))
        self.assertEqual(set(self.alice), {'id', 'display_name', 'email'})

    def test_passwords_use_independent_salts_and_pbkdf2(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(
                'SELECT password_hash, password_salt, password_iterations FROM users ORDER BY id'
            ).fetchall()
        self.assertNotEqual(rows[0][0], rows[1][0])
        self.assertNotEqual(rows[0][1], rows[1][1])
        for digest, salt, iterations in rows:
            self.assertEqual(iterations, 600_000)
            self.assertEqual(iterations, PASSWORD_ITERATIONS)
            self.assertEqual(len(salt), 32)
            self.assertEqual(digest, hashlib.pbkdf2_hmac('sha256', self.password.encode(), salt, iterations))
            self.assertNotEqual(digest, self.password.encode())
        self.assertNotIn(self.password.encode(), self.db_path.read_bytes())

    def test_duplicate_email_and_invalid_registration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.store.create_user('Another Alice', 'ALICE@example.com', self.password)
        cases = [('', 'valid@example.com', self.password),
                 ('A' * 101, 'valid@example.com', self.password),
                 ('Alice\x00', 'valid@example.com', self.password),
                 ('Alice', 'not-an-email', self.password),
                 ('Alice', None, self.password),
                 ('Alice', 'valid@example.com', 'short'),
                 ('Alice', 'valid@example.com', ' ' * 20),
                 ('Alice', 'valid@example.com', b'a-byte-password'),
                 ('Alice', 'valid@example.com', 'x' * 1025)]
        for args in cases:
            with self.subTest(args=args[:2]), self.assertRaises(ValueError):
                self.store.create_user(*args)

    def test_datasets_preserve_exact_bytes_and_timezone_metadata(self):
        raw = b'\xef\xbb\xbfdescription,amount\r\n"first\nsecond",100\r\n'
        saved = self.store.save_dataset(self.alice['id'], ' Upload.csv ', raw)
        self.assertEqual(saved['name'], 'Upload.csv')
        self.assertEqual(saved['row_count'], 1)
        self.assertEqual(saved['size_bytes'], len(raw))
        self.assertIsNotNone(datetime.fromisoformat(saved['created_at']).tzinfo)
        loaded = self.store.get_dataset(self.alice['id'], saved['id'])
        self.assertEqual(loaded['csv_bytes'], raw)
        self.assertNotIn('csv_bytes', self.store.list_datasets(self.alice['id'])[0])

    def test_dataset_and_run_ownership_are_enforced(self):
        saved = self.dataset()
        record = self.run_record(saved['id'])
        self.assertIsNone(self.store.get_dataset(self.bob['id'], saved['id']))
        self.assertEqual(self.store.list_datasets(self.bob['id']), [])
        self.assertIsNone(self.store.get_run(self.bob['id'], record['id']))
        self.assertEqual(self.store.list_runs(self.bob['id'], saved['id']), [])
        with self.assertRaisesRegex(ValueError, 'not available'):
            self.run_record(saved['id'], self.bob)
        self.assertEqual(self.store.list_runs(self.bob['id']), [])

    def test_result_history_and_dataset_filters(self):
        first, second = self.dataset(), self.dataset(name='second.csv')
        first_run = self.run_record(first['id'])
        second_run = self.run_record(second['id'])
        single = self.run_record(kind='single')
        self.assertEqual([row['id'] for row in self.store.list_runs(self.alice['id'])],
                         [single['id'], second_run['id'], first_run['id']])
        self.assertEqual(self.store.list_runs(self.alice['id'], first['id']), [first_run])
        self.assertNotIn('result_csv', first_run)
        self.assertEqual(self.store.get_run(self.alice['id'], single['id'])['result_csv'], self.result_csv)
        self.assertEqual(first_run['summary'], {'total': 2, 'fraud': 1})
        self.assertIsNone(single['dataset_id'])

    def test_active_workspace_is_owned_and_persists_across_restarts(self):
        first, second = self.dataset(), self.dataset(name='second.csv')
        run = self.run_record(first['id'])
        self.assertIsNone(self.store.get_active_dataset(self.alice['id']))
        self.store.set_active_dataset(self.alice['id'], first['id'])
        with self.assertRaisesRegex(ValueError, 'not available'):
            self.store.set_active_dataset(self.bob['id'], first['id'])
        self.store.close()
        self.store = WorkspaceStore(self.db_path)
        self.assertEqual(self.store.authenticate('alice@example.com', self.password), self.alice)
        self.assertEqual(self.store.get_active_dataset(self.alice['id'])['id'], first['id'])
        self.assertEqual(len(self.store.list_datasets(self.alice['id'])), 2)
        self.assertEqual(self.store.get_run(self.alice['id'], run['id'])['result_csv'], self.result_csv)
        self.store.set_active_dataset(self.alice['id'], second['id'])
        self.assertEqual(self.store.get_active_dataset(self.alice['id'])['id'], second['id'])
        self.store.set_active_dataset(self.alice['id'], None)
        self.assertIsNone(self.store.get_active_dataset(self.alice['id']))

    def test_database_foreign_keys_also_block_cross_account_references(self):
        saved = self.dataset()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute('PRAGMA foreign_keys = ON')
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    '''INSERT INTO runs
                       (user_id, dataset_id, model_name, result_csv, summary, kind, size_bytes, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (self.bob['id'], saved['id'], 'Model', self.result_csv, '{}', 'batch', 1, '2026-10-19T00:00:00+00:00'),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute('INSERT INTO user_preferences VALUES (?, ?)', (self.bob['id'], saved['id']))

    def test_untrusted_text_is_stored_as_data(self):
        sql_text = "transactions'); DROP TABLE users; --"
        saved = self.dataset(name=sql_text)
        record = self.store.save_run(self.alice['id'], saved['id'], sql_text, self.result_csv,
                                     {'description': sql_text})
        self.assertEqual(self.store.get_dataset(self.alice['id'], saved['id'])['name'], sql_text)
        self.assertEqual(self.store.get_run(self.alice['id'], record['id'])['summary']['description'], sql_text)
        self.assertEqual(self.store.authenticate('bob@example.com', self.password), self.bob)
        with self.assertRaises(ValueError):
            self.store.get_dataset(self.alice['id'], '1 OR 1=1')

    def test_invalid_csv_payloads_leave_no_partial_dataset(self):
        invalid = ['', bytearray(self.csv_bytes), b'', b'amount,type\n', b'a,a\n1,2\n',
                   b'a,\n1,2\n', b'a,b\n1,2,3\n', b'a\n"unclosed\n',
                   b'a\n\xff\n', b'a\n\x00\n', b'x' * (MAX_CSV_BYTES + 1)]
        for payload in invalid:
            with self.subTest(payload_type=type(payload).__name__, size=len(payload)), self.assertRaises(ValueError):
                self.store.save_dataset(self.alice['id'], 'invalid.csv', payload)
        self.assertEqual(self.store.list_datasets(self.alice['id']), [])

    def test_invalid_runs_and_identifiers_leave_no_partial_run(self):
        saved = self.dataset()
        for user_id in (True, -1, 0, '1', None):
            with self.subTest(user_id=user_id), self.assertRaises(ValueError):
                self.store.list_runs(user_id)
        cases = [{'summary': {'score': float('nan')}}, {'summary': {'object': object()}},
                 {'summary': []}, {'summary': {'oversize': 'x' * (128 * 1024)}},
                 {'kind': 'unknown'}, {'model_name': ''}, {'result_csv': 'not bytes'},
                 {'dataset_id': 99999}, {'user_id': 99999}]
        for overrides in cases:
            args = {'user_id': self.alice['id'], 'dataset_id': saved['id'],
                    'model_name': 'Random Forest', 'result_csv': self.result_csv, 'summary': {'total': 2}}
            args.update(overrides)
            with self.subTest(overrides=list(overrides)), self.assertRaises(ValueError):
                self.store.save_run(**args)
        with self.assertRaisesRegex(ValueError, 'no longer exists'):
            self.store.save_dataset(99999, 'missing.csv', self.csv_bytes)
        self.assertEqual(self.store.list_runs(self.alice['id']), [])

    def test_in_memory_store_keeps_state_for_its_lifetime(self):
        with WorkspaceStore(':memory:') as memory_store:
            user = memory_store.create_user('Demo User', 'demo@example.com', self.password)
            saved = memory_store.save_dataset(user['id'], 'transactions.csv', self.csv_bytes)
            self.assertEqual(memory_store.get_dataset(user['id'], saved['id'])['csv_bytes'], self.csv_bytes)

    def test_transaction_review_and_audit_history(self):
        dataset = self.dataset()
        run = self.run_record(dataset['id'])
        self.assertIsNone(self.store.get_review(self.alice['id'], run['id'], 1))
        
        # Save review
        review = self.store.save_review(self.alice['id'], run['id'], 1, 'Pending',
                                       'Initial suspicious pattern detected', self.alice['display_name'])
        self.assertEqual(review['status'], 'Pending')
        self.assertEqual(review['notes'], 'Initial suspicious pattern detected')
        
        # Read back
        loaded = self.store.get_review(self.alice['id'], run['id'], 1)
        self.assertEqual(loaded['status'], 'Pending')
        self.assertEqual(loaded['notes'], 'Initial suspicious pattern detected')
        
        # Update review
        self.store.save_review(self.alice['id'], run['id'], 1, 'Escalated',
                              'Escalated to senior analyst', self.alice['display_name'])
        updated = self.store.get_review(self.alice['id'], run['id'], 1)
        self.assertEqual(updated['status'], 'Escalated')
        self.assertEqual(updated['notes'], 'Escalated to senior analyst')
        
        # Check audit trail
        audit = self.store.list_review_audit(self.alice['id'], run['id'], 1)
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[0]['status'], 'Pending')
        self.assertEqual(audit[1]['status'], 'Escalated')
        
        # Check isolation: Bob cannot view or update Alice's reviews
        self.assertIsNone(self.store.get_review(self.bob['id'], run['id'], 1))
        with self.assertRaises(ValueError):
            self.store.save_review(self.bob['id'], run['id'], 1, 'Reviewed', 'Notes', self.bob['display_name'])



if __name__ == '__main__':
    unittest.main()
