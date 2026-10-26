"""Local accounts and account-owned datasets/prediction history.

This SQLite store is intended for the local project demonstration. Accounts are
not email-verified and the database is not encrypted at rest. Passwords are
salted and hashed; raw passwords are never written to the database.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from .dataset_validation import MAX_RESULT_BYTES, MAX_UPLOAD_BYTES
except ImportError:
    from dataset_validation import MAX_RESULT_BYTES, MAX_UPLOAD_BYTES

PASSWORD_ITERATIONS = 600_000
MAX_CSV_BYTES = MAX_UPLOAD_BYTES  # Compatibility for existing callers.
MAX_SUMMARY_BYTES = 128 * 1024
REVIEW_STATUSES = ['Pending', 'Reviewed', 'Escalated']
MAX_REVIEW_NOTES = 5000


class WorkspaceStore:
    """Thread-safe SQLite persistence for local accounts and their work."""

    def __init__(self, db_path):
        if str(db_path) != ':memory:':
            Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            db_path = str(Path(db_path).expanduser().resolve())
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute('PRAGMA foreign_keys = ON')
        self._connection.execute('PRAGMA busy_timeout = 10000')
        self._connection.execute('PRAGMA journal_mode = WAL')
        with self._connection:
            self._connection.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash BLOB NOT NULL,
                    password_salt BLOB NOT NULL,
                    password_iterations INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    name TEXT NOT NULL,
                    csv_bytes BLOB NOT NULL,
                    row_count INTEGER NOT NULL CHECK(row_count >= 0),
                    size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, id)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    dataset_id INTEGER,
                    model_name TEXT NOT NULL,
                    result_csv BLOB NOT NULL,
                    summary TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('batch', 'single')),
                    size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id, dataset_id) REFERENCES datasets(user_id, id)
                );
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id),
                    active_dataset_id INTEGER,
                    FOREIGN KEY(user_id, active_dataset_id) REFERENCES datasets(user_id, id)
                );
                CREATE INDEX IF NOT EXISTS datasets_user_idx ON datasets(user_id, id DESC);
                CREATE INDEX IF NOT EXISTS runs_user_idx ON runs(user_id, id DESC);
                CREATE INDEX IF NOT EXISTS runs_dataset_idx ON runs(user_id, dataset_id, id DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS runs_owner_id_idx ON runs(user_id, id);
                CREATE TABLE IF NOT EXISTS transaction_reviews (
                    user_id INTEGER NOT NULL,
                    run_id INTEGER NOT NULL,
                    source_row INTEGER NOT NULL CHECK(source_row >= 1),
                    status TEXT NOT NULL CHECK(status IN ('Pending', 'Reviewed', 'Escalated')),
                    notes TEXT NOT NULL CHECK(length(notes) <= 5000),
                    reviewer_user_id INTEGER NOT NULL REFERENCES users(id),
                    reviewer_display_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, run_id, source_row),
                    CHECK(reviewer_user_id = user_id),
                    FOREIGN KEY(user_id, run_id) REFERENCES runs(user_id, id)
                );
                CREATE TABLE IF NOT EXISTS review_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    run_id INTEGER NOT NULL,
                    source_row INTEGER NOT NULL CHECK(source_row >= 1),
                    status TEXT NOT NULL CHECK(status IN ('Pending', 'Reviewed', 'Escalated')),
                    notes TEXT NOT NULL CHECK(length(notes) <= 5000),
                    reviewer_user_id INTEGER NOT NULL REFERENCES users(id),
                    reviewer_display_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(reviewer_user_id = user_id),
                    FOREIGN KEY(user_id, run_id) REFERENCES runs(user_id, id)
                );
                CREATE INDEX IF NOT EXISTS review_audit_row_idx
                    ON review_audit(user_id, run_id, source_row, id);
                CREATE TRIGGER IF NOT EXISTS review_audit_no_update
                    BEFORE UPDATE ON review_audit
                    BEGIN SELECT RAISE(ABORT, 'Review audit history is immutable.'); END;
                CREATE TRIGGER IF NOT EXISTS review_audit_no_delete
                    BEFORE DELETE ON review_audit
                    BEGIN SELECT RAISE(ABORT, 'Review audit history is immutable.'); END;

                CREATE TABLE IF NOT EXISTS investigation_cases (
                    case_id TEXT PRIMARY KEY,
                    event_id TEXT UNIQUE NOT NULL,
                    transaction_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    txn_type TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_tier TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    assigned_analyst TEXT NOT NULL DEFAULT 'Unassigned',
                    status TEXT NOT NULL CHECK(status IN ('OPEN', 'IN_REVIEW', 'ESCALATED', 'RESOLVED')),
                    outcome TEXT NOT NULL CHECK(outcome IN ('PENDING', 'CONFIRMED_FRAUD', 'FALSE_ALARM', 'SUSPICIOUS_CLEARED')),
                    notes TEXT NOT NULL DEFAULT '',
                    explanation_json TEXT NOT NULL DEFAULT '{}',
                    model_version TEXT NOT NULL DEFAULT 'v1.0.0-xgb-consensus',
                    preprocessor_version TEXT NOT NULL DEFAULT 'v1.0.0-robust',
                    decision_threshold REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cases_event_id ON investigation_cases(event_id);
                CREATE INDEX IF NOT EXISTS idx_cases_status ON investigation_cases(status);

                CREATE TABLE IF NOT EXISTS case_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES investigation_cases(case_id),
                    changed_by TEXT NOT NULL,
                    action TEXT NOT NULL,
                    old_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_case_audit ON case_audit_log(case_id, id ASC);

                CREATE TABLE IF NOT EXISTS api_rate_limits (
                    client_key TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rate_limits_key_ts ON api_rate_limits(client_key, timestamp);
            ''')
            # Additive migration: old account/dataset/result rows stay intact.
            columns = {row['name'] for row in self._connection.execute('PRAGMA table_info(runs)')}
            if 'row_count' not in columns:
                self._connection.execute('ALTER TABLE runs ADD COLUMN row_count INTEGER')

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec='microseconds')

    @staticmethod
    def _text(value, label, max_length):
        if not isinstance(value, str):
            raise ValueError(f'{label} must be text.')
        value = value.strip()
        if not value or len(value) > max_length:
            raise ValueError(f'{label} must contain between 1 and {max_length} characters.')
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f'{label} cannot contain control characters.')
        return value

    @classmethod
    def _email(cls, email):
        email = cls._text(email, 'Email', 254).casefold()
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
            raise ValueError('Enter a valid email address, such as name@example.com.')
        return email

    @staticmethod
    def _id(value, label):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{label} must be a positive integer.')
        return value

    @staticmethod
    def _password(password, creating=False):
        if not isinstance(password, str):
            raise ValueError('Password must be text.')
        if len(password) > 1024:
            raise ValueError('Password must be at most 1024 characters.')
        if creating and (len(password) < 12 or not password.strip()):
            raise ValueError('Password must contain at least 12 characters and cannot be only spaces.')
        return password.encode('utf-8')

    @staticmethod
    def _csv_info(payload, label, max_bytes=MAX_UPLOAD_BYTES):
        if not isinstance(payload, bytes):
            raise ValueError(f'{label} must be CSV file bytes.')
        if not payload or len(payload) > max_bytes:
            raise ValueError(f'{label} must be a non-empty CSV file no larger than {max_bytes // (1024 * 1024)} MB.')
        try:
            decoded = payload.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError(f'{label} must use UTF-8 encoding.') from exc
        if '\x00' in decoded:
            raise ValueError(f'{label} contains invalid null characters.')
        try:
            reader = csv.reader(io.StringIO(decoded, newline=''), strict=True)
            header = next(reader, None)
            if not header or any(not column.strip() for column in header):
                raise ValueError(f'{label} must include non-empty column headers.')
            if len(set(column.strip() for column in header)) != len(header):
                raise ValueError(f'{label} contains duplicate column headers.')
            row_count = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(f'{label} contains a row with an unexpected number of columns.')
                row_count += 1
        except csv.Error as exc:
            raise ValueError(f'{label} is not a valid CSV file: {exc}.') from exc
        if not row_count:
            raise ValueError(f'{label} must include at least one data row.')
        return row_count, len(payload)

    def _require_user(self, user_id):
        if self._connection.execute('SELECT 1 FROM users WHERE id = ?', (user_id,)).fetchone() is None:
            raise ValueError('This account no longer exists. Please sign in again.')

    def create_user(self, display_name, email, password):
        display_name = self._text(display_name, 'Display name', 100)
        email = self._email(email)
        password_bytes = self._password(password, creating=True)
        salt = secrets.token_bytes(32)
        digest = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, PASSWORD_ITERATIONS)
        with self._lock, self._connection:
            try:
                cursor = self._connection.execute(
                    '''INSERT INTO users
                       (display_name, email, password_hash, password_salt, password_iterations, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (display_name, email, digest, salt, PASSWORD_ITERATIONS, self._now()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError('An account already exists for this email address. Please sign in.') from exc
            return {'id': cursor.lastrowid, 'display_name': display_name, 'email': email}

    def authenticate(self, email, password):
        email = self._email(email)
        password_bytes = self._password(password)
        with self._lock:
            user = self._connection.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        # Perform the same expensive hash for unknown accounts to avoid a cheap
        # account-existence timing signal. Authentication errors stay generic.
        salt = bytes(user['password_salt']) if user else bytes(32)
        iterations = user['password_iterations'] if user else PASSWORD_ITERATIONS
        expected = bytes(user['password_hash']) if user else bytes(32)
        actual = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, iterations)
        if not hmac.compare_digest(actual, expected) or user is None:
            return None
        return {key: user[key] for key in ('id', 'display_name', 'email')}

    def save_dataset(self, user_id, name, csv_bytes):
        user_id = self._id(user_id, 'User ID')
        name = self._text(name, 'Dataset name', 255)
        row_count, size_bytes = self._csv_info(csv_bytes, 'Dataset')
        created_at = self._now()
        with self._lock, self._connection:
            self._require_user(user_id)
            cursor = self._connection.execute(
                '''INSERT INTO datasets (user_id, name, csv_bytes, row_count, size_bytes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (user_id, name, csv_bytes, row_count, size_bytes, created_at),
            )
            return {'id': cursor.lastrowid, 'name': name, 'row_count': row_count,
                    'size_bytes': size_bytes, 'created_at': created_at}

    def list_datasets(self, user_id):
        user_id = self._id(user_id, 'User ID')
        with self._lock:
            rows = self._connection.execute(
                '''SELECT id, name, row_count, size_bytes, created_at FROM datasets
                   WHERE user_id = ? ORDER BY id DESC''', (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_dataset(self, user_id, dataset_id):
        user_id = self._id(user_id, 'User ID')
        dataset_id = self._id(dataset_id, 'Dataset ID')
        with self._lock:
            row = self._connection.execute(
                '''SELECT id, name, row_count, size_bytes, created_at, csv_bytes FROM datasets
                   WHERE user_id = ? AND id = ?''', (user_id, dataset_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_active_dataset(self, user_id):
        user_id = self._id(user_id, 'User ID')
        with self._lock:
            row = self._connection.execute(
                '''SELECT d.id, d.name, d.row_count, d.size_bytes, d.created_at, d.csv_bytes
                   FROM datasets AS d JOIN user_preferences AS p
                     ON p.user_id = d.user_id AND p.active_dataset_id = d.id
                   WHERE p.user_id = ?''', (user_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def set_active_dataset(self, user_id, dataset_id):
        user_id = self._id(user_id, 'User ID')
        if dataset_id is not None:
            dataset_id = self._id(dataset_id, 'Dataset ID')
        with self._lock, self._connection:
            self._require_user(user_id)
            if dataset_id is not None and self._connection.execute(
                'SELECT 1 FROM datasets WHERE id = ? AND user_id = ?', (dataset_id, user_id),
            ).fetchone() is None:
                raise ValueError('The selected dataset is not available in your account.')
            self._connection.execute(
                '''INSERT INTO user_preferences (user_id, active_dataset_id) VALUES (?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET active_dataset_id = excluded.active_dataset_id''',
                (user_id, dataset_id),
            )

    def save_run(self, user_id, dataset_id, model_name, result_csv, summary, kind='batch'):
        user_id = self._id(user_id, 'User ID')
        if dataset_id is not None:
            dataset_id = self._id(dataset_id, 'Dataset ID')
        model_name = self._text(model_name, 'Model name', 100)
        if kind not in ('batch', 'single'):
            raise ValueError('Run kind must be either batch or single.')
        _, size_bytes = self._csv_info(result_csv, 'Prediction results')
        if not isinstance(summary, dict):
            raise ValueError('Prediction summary must be a dictionary.')
        try:
            summary_json = json.dumps(summary, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Prediction summary must contain valid JSON values.') from exc
        if len(summary_json.encode('utf-8')) > MAX_SUMMARY_BYTES:
            raise ValueError('Prediction summary must be no larger than 128 KB.')
        created_at = self._now()
        with self._lock, self._connection:
            self._require_user(user_id)
            if dataset_id is not None and self._connection.execute(
                'SELECT 1 FROM datasets WHERE id = ? AND user_id = ?', (dataset_id, user_id),
            ).fetchone() is None:
                raise ValueError('The selected dataset is not available in your account.')
            cursor = self._connection.execute(
                '''INSERT INTO runs
                   (user_id, dataset_id, model_name, result_csv, summary, kind, size_bytes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (user_id, dataset_id, model_name, result_csv, summary_json, kind, size_bytes, created_at),
            )
            return {'id': cursor.lastrowid, 'dataset_id': dataset_id, 'model_name': model_name,
                    'kind': kind, 'summary': json.loads(summary_json),
                    'size_bytes': size_bytes, 'created_at': created_at}

    def list_runs(self, user_id, dataset_id=None):
        user_id = self._id(user_id, 'User ID')
        sql = '''SELECT id, dataset_id, model_name, kind, summary, size_bytes, created_at
                 FROM runs WHERE user_id = ?'''
        parameters = [user_id]
        if dataset_id is not None:
            dataset_id = self._id(dataset_id, 'Dataset ID')
            sql += ' AND dataset_id = ?'
            parameters.append(dataset_id)
        sql += ' ORDER BY id DESC'
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return [self._run_dict(row) for row in rows]

    def get_run(self, user_id, run_id):
        user_id = self._id(user_id, 'User ID')
        run_id = self._id(run_id, 'Run ID')
        with self._lock:
            row = self._connection.execute(
                '''SELECT id, dataset_id, model_name, kind, summary, size_bytes, created_at, result_csv
                   FROM runs WHERE user_id = ? AND id = ?''', (user_id, run_id),
            ).fetchone()
        return self._run_dict(row) if row is not None else None

    def save_review(self, user_id, run_id, source_row, status, notes, reviewer_display_name):
        user_id = self._id(user_id, 'User ID')
        run_id = self._id(run_id, 'Run ID')
        source_row = self._id(source_row, 'Source row')
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Review status must be one of {', '.join(REVIEW_STATUSES)}.")
        if not isinstance(notes, str):
            raise ValueError("Review notes must be text.")
        if len(notes) > MAX_REVIEW_NOTES:
            raise ValueError(f"Review notes cannot exceed {MAX_REVIEW_NOTES} characters.")
        reviewer_display_name = self._text(reviewer_display_name, 'Reviewer name', 100)
        now = self._now()
        with self._lock, self._connection:
            self._require_user(user_id)
            if self._connection.execute(
                'SELECT 1 FROM runs WHERE id = ? AND user_id = ?', (run_id, user_id)
            ).fetchone() is None:
                raise ValueError('The selected analysis run is not available in your account.')
            self._connection.execute(
                '''INSERT INTO transaction_reviews
                   (user_id, run_id, source_row, status, notes, reviewer_user_id, reviewer_display_name, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, run_id, source_row) DO UPDATE SET
                   status = excluded.status,
                   notes = excluded.notes,
                   reviewer_display_name = excluded.reviewer_display_name,
                   updated_at = excluded.updated_at''',
                (user_id, run_id, source_row, status, notes, user_id, reviewer_display_name, now)
            )
            self._connection.execute(
                '''INSERT INTO review_audit
                   (user_id, run_id, source_row, status, notes, reviewer_user_id, reviewer_display_name, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (user_id, run_id, source_row, status, notes, user_id, reviewer_display_name, now)
            )
            return {
                'user_id': user_id,
                'run_id': run_id,
                'source_row': source_row,
                'status': status,
                'notes': notes,
                'reviewer_display_name': reviewer_display_name,
                'updated_at': now
            }

    def get_review(self, user_id, run_id, source_row):
        user_id = self._id(user_id, 'User ID')
        run_id = self._id(run_id, 'Run ID')
        source_row = self._id(source_row, 'Source row')
        with self._lock:
            row = self._connection.execute(
                '''SELECT user_id, run_id, source_row, status, notes, reviewer_display_name, updated_at
                   FROM transaction_reviews
                   WHERE user_id = ? AND run_id = ? AND source_row = ?''',
                (user_id, run_id, source_row)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_reviews(self, user_id, run_id=None):
        user_id = self._id(user_id, 'User ID')
        sql = '''SELECT user_id, run_id, source_row, status, notes, reviewer_display_name, updated_at
                 FROM transaction_reviews WHERE user_id = ?'''
        parameters = [user_id]
        if run_id is not None:
            run_id = self._id(run_id, 'Run ID')
            sql += ' AND run_id = ?'
            parameters.append(run_id)
        sql += ' ORDER BY updated_at DESC'
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_review_audit(self, user_id, run_id, source_row):
        user_id = self._id(user_id, 'User ID')
        run_id = self._id(run_id, 'Run ID')
        source_row = self._id(source_row, 'Source row')
        with self._lock:
            rows = self._connection.execute(
                '''SELECT id, user_id, run_id, source_row, status, notes, reviewer_display_name, updated_at
                   FROM review_audit
                   WHERE user_id = ? AND run_id = ? AND source_row = ?
                   ORDER BY id ASC''',
                (user_id, run_id, source_row)
            ).fetchall()
        return [dict(row) for row in rows]


    @staticmethod
    def _run_dict(row):
        result = dict(row)
        result['summary'] = json.loads(result['summary'])
        return result

    # -------------------------------------------------------------
    # FRAUD INVESTIGATION QUEUE & IDEMPOTENT CASE MANAGEMENT
    # -------------------------------------------------------------
    def create_or_get_case(
        self,
        event_id: str,
        transaction_id: str,
        amount: float,
        txn_type: str,
        risk_score: float,
        risk_tier: str,
        decision: str,
        assigned_analyst: str = 'Unassigned',
        status: str = 'OPEN',
        outcome: str = 'PENDING',
        notes: str = '',
        explanation: dict = None,
        model_version: str = 'v1.0.0-xgb-consensus',
        preprocessor_version: str = 'v1.0.0-robust',
        decision_threshold: float = 0.5,
    ) -> tuple[dict, bool]:
        """
        Idempotently create or retrieve an investigation case for an incoming event.
        Guarantees that replayed Kafka events or duplicate submissions create ZERO duplicate cases.
        Returns: (case_dict, was_created: bool)
        """
        event_id = str(event_id).strip()
        transaction_id = str(transaction_id).strip()
        now = self._now()
        explanation_str = json.dumps(explanation or {})

        with self._lock, self._connection:
            # Check existing event
            existing = self._connection.execute(
                'SELECT * FROM investigation_cases WHERE event_id = ?', (event_id,)
            ).fetchone()
            if existing is not None:
                return dict(existing), False

            case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

            self._connection.execute(
                '''INSERT INTO investigation_cases
                   (case_id, event_id, transaction_id, amount, txn_type, risk_score,
                    risk_tier, decision, assigned_analyst, status, outcome, notes,
                    explanation_json, model_version, preprocessor_version, decision_threshold,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (case_id, event_id, transaction_id, float(amount), str(txn_type),
                 float(risk_score), str(risk_tier), str(decision), str(assigned_analyst),
                 str(status), str(outcome), str(notes), explanation_str,
                 str(model_version), str(preprocessor_version), float(decision_threshold),
                 now, now),
            )

            # Record creation in immutable audit trail
            self._connection.execute(
                '''INSERT INTO case_audit_log
                   (case_id, changed_by, action, old_state, new_state, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (case_id, 'SYSTEM_INGESTION', 'CASE_CREATED', '{}',
                 json.dumps({'status': status, 'outcome': outcome, 'assigned_analyst': assigned_analyst}),
                 now),
            )

            created_row = self._connection.execute(
                'SELECT * FROM investigation_cases WHERE case_id = ?', (case_id,)
            ).fetchone()
            return dict(created_row), True

    def update_case(
        self,
        case_id: str,
        changed_by: str,
        status: str = None,
        outcome: str = None,
        notes: str = None,
        assigned_analyst: str = None,
    ) -> dict:
        """Update an investigation case and record an immutable audit log entry."""
        case_id = str(case_id).strip()
        changed_by = self._text(changed_by, 'Changed by', 100)
        now = self._now()

        with self._lock, self._connection:
            row = self._connection.execute(
                'SELECT * FROM investigation_cases WHERE case_id = ?', (case_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Investigation case '{case_id}' does not exist.")

            old_state = {
                'status': row['status'],
                'outcome': row['outcome'],
                'notes': row['notes'],
                'assigned_analyst': row['assigned_analyst'],
            }

            new_status = str(status) if status is not None else row['status']
            new_outcome = str(outcome) if outcome is not None else row['outcome']
            new_notes = str(notes) if notes is not None else row['notes']
            new_analyst = str(assigned_analyst) if assigned_analyst is not None else row['assigned_analyst']

            if new_status not in ('OPEN', 'IN_REVIEW', 'ESCALATED', 'RESOLVED'):
                raise ValueError(f"Invalid status: {new_status}")
            if new_outcome not in ('PENDING', 'CONFIRMED_FRAUD', 'FALSE_ALARM', 'SUSPICIOUS_CLEARED'):
                raise ValueError(f"Invalid outcome: {new_outcome}")

            self._connection.execute(
                '''UPDATE investigation_cases
                   SET status = ?, outcome = ?, notes = ?, assigned_analyst = ?, updated_at = ?
                   WHERE case_id = ?''',
                (new_status, new_outcome, new_notes, new_analyst, now, case_id),
            )

            new_state = {
                'status': new_status,
                'outcome': new_outcome,
                'notes': new_notes,
                'assigned_analyst': new_analyst,
            }

            self._connection.execute(
                '''INSERT INTO case_audit_log
                   (case_id, changed_by, action, old_state, new_state, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (case_id, changed_by, 'CASE_UPDATED', json.dumps(old_state), json.dumps(new_state), now),
            )

            updated_row = self._connection.execute(
                'SELECT * FROM investigation_cases WHERE case_id = ?', (case_id,)
            ).fetchone()
            return dict(updated_row)

    def get_case(self, case_id: str) -> dict | None:
        """Fetch a specific investigation case by case_id."""
        with self._lock:
            row = self._connection.execute(
                'SELECT * FROM investigation_cases WHERE case_id = ?', (str(case_id).strip(),)
            ).fetchone()
            return dict(row) if row is not None else None

    def get_case_by_event(self, event_id: str) -> dict | None:
        """Fetch an investigation case by unique Kafka event_id."""
        with self._lock:
            row = self._connection.execute(
                'SELECT * FROM investigation_cases WHERE event_id = ?', (str(event_id).strip(),)
            ).fetchone()
            return dict(row) if row is not None else None

    def list_cases(self, status: str = None, outcome: str = None, limit: int = 100) -> list[dict]:
        """List investigation cases sorted by most recent first."""
        sql = 'SELECT * FROM investigation_cases WHERE 1=1'
        params = []
        if status:
            sql += ' AND status = ?'
            params.append(status)
        if outcome:
            sql += ' AND outcome = ?'
            params.append(outcome)
        sql += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)

        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_case_audit_log(self, case_id: str) -> list[dict]:
        """Fetch chronological audit trail for a case."""
        with self._lock:
            rows = self._connection.execute(
                'SELECT * FROM case_audit_log WHERE case_id = ? ORDER BY id ASC',
                (str(case_id).strip(),)
            ).fetchall()
            return [dict(r) for r in rows]

    def check_and_record_rate_limit(self, client_key: str, limit: int = 120, window_seconds: float = 60.0) -> bool:
        """Process-safe sliding window rate limiter shared across all workers in SQLite."""
        now = time.time()
        cutoff = now - window_seconds
        with self._lock, self._connection:
            self._connection.execute('DELETE FROM api_rate_limits WHERE timestamp < ?', (cutoff,))
            cur = self._connection.execute(
                'SELECT COUNT(*) FROM api_rate_limits WHERE client_key = ? AND timestamp >= ?',
                (client_key, cutoff)
            )
            count = cur.fetchone()[0]
            if count >= limit:
                return False
            self._connection.execute(
                'INSERT INTO api_rate_limits (client_key, timestamp) VALUES (?, ?)',
                (client_key, now)
            )
            return True

    def reset_rate_limits(self, client_key: str | None = None):
        """Reset rate limit entries, optionally for a specific client."""
        with self._lock, self._connection:
            if client_key:
                self._connection.execute('DELETE FROM api_rate_limits WHERE client_key = ?', (client_key,))
            else:
                self._connection.execute('DELETE FROM api_rate_limits')

    def ping(self) -> bool:
        """Verify SQLite database connectivity and alive state."""
        with self._lock:
            res = self._connection.execute('SELECT 1').fetchone()
            return res is not None

    def close(self):
        with self._lock:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
