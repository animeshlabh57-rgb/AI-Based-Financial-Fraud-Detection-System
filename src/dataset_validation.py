"""Row-level input checks without dropping or changing uploaded transactions.

Only required model inputs affect validity. Labels and other metadata do not.
Missing input cells are warnings because saved training statistics can impute
them; an entirely empty transaction is invalid. Duplicate inputs are retained.
"""

from __future__ import annotations


UPLOAD_LIMIT_MB = 25
MAX_UPLOAD_BYTES = UPLOAD_LIMIT_MB * 1024 * 1024
MAX_RESULT_BYTES = 100 * 1024 * 1024


def normalize_missing_values(df):
    """Return a copy with blank raw-input strings converted to missing values.

    Call this before inference on the rows selected by ``valid_mask``. The
    original frame and all metadata columns remain untouched.
    """
    import numpy as np
    import pandas as pd
    import config

    if not isinstance(df, pd.DataFrame):
        raise ValueError('Provide transaction data as a pandas DataFrame.')
    if not df.columns.is_unique:
        raise ValueError('Transaction column names must be unique.')
    result = df.copy()
    for column in config.RAW_FEATURE_COLUMNS:
        if column in result:
            blanks = result[column].astype('string').str.strip().eq('').fillna(False)
            if blanks.any():
                result[column] = result[column].mask(blanks, np.nan)
    return result


def validate_dataset(df):
    """Return positional row validity, readable issues, and dataset counts.

    ``Source_Row`` is one-based input position; it is independent of potentially
    repeated pandas indexes or transaction IDs. The returned mask has the same
    index and order as the source. This function never drops or imputes rows.
    Duplicate counts concern repeated required model inputs, excluding labels
    and identifiers. ``can_analyze`` means at least one row can be analysed.
    """
    import numpy as np
    import pandas as pd
    import config

    if not isinstance(df, pd.DataFrame):
        raise ValueError('Provide transaction data as a pandas DataFrame.')
    count = len(df)
    required = list(config.RAW_FEATURE_COLUMNS)
    missing_columns = [column for column in required if column not in df.columns]
    valid = np.ones(count, dtype=bool)
    issues = []

    def add(mask, column, message, invalid=False):
        positions = np.flatnonzero(np.asarray(mask, dtype=bool))
        issues.extend({'Source_Row': int(position) + 1, 'Column': column, 'Issue': message}
                      for position in positions)
        if invalid:
            valid[positions] = False

    available = [column for column in required if column in df.columns]
    missing_values = 0
    duplicate_rows = 0
    if not df.columns.is_unique:
        for column in df.columns[df.columns.duplicated()].unique():
            add(np.ones(count, dtype=bool), str(column),
                'Duplicate column name; rename the column before analysis.', invalid=True)
    else:
        normalized = normalize_missing_values(df)
        raw = normalized[available]
        missing_values = int(raw.isna().sum().sum())
        if available:
            duplicates = raw.duplicated(keep='first').to_numpy()
            duplicate_rows = int(duplicates.sum())
            add(duplicates, '(row)', 'Warning: duplicate transaction inputs; this row is preserved.')
        for column in available:
            missing = normalized[column].isna().to_numpy()
            add(missing, column,
                'Warning: missing value; saved training statistics will impute this input.')
            if column == 'type':
                types = normalized[column].astype('string').str.strip().str.upper()
                unsupported = (~types.isin(config.TRANSACTION_TYPES)).to_numpy() & ~missing
                add(unsupported, column, 'Unsupported transaction type; use '
                    + ', '.join(config.TRANSACTION_TYPES) + '.', invalid=True)
                continue
            numeric = pd.to_numeric(normalized[column], errors='coerce')
            numbers = numeric.to_numpy(dtype=float, na_value=np.nan)
            not_numeric = np.isnan(numbers) & ~missing
            add(not_numeric, column, 'Must contain a numeric value.', invalid=True)
            non_finite = np.isinf(numbers)
            add(non_finite, column, 'Must be finite; infinity is not allowed.', invalid=True)
            finite = np.isfinite(numbers)
            if column == 'step':
                invalid_step = finite & ((numbers < 1) | (numbers > 744) | (numbers != np.floor(numbers)))
                add(invalid_step, column, 'Step must be a whole number from 1 to 744.', invalid=True)
            else:
                add(finite & (numbers < 0), column, 'Amount and balances cannot be negative.', invalid=True)
        if not missing_columns and available:
            add(raw.isna().all(axis=1).to_numpy(), '(row)',
                'All required transaction inputs are missing; provide a transaction before analysis.', invalid=True)

    for column in missing_columns:
        add(np.ones(count, dtype=bool), column,
            f'Missing required column: {column}. Add this column before analysis.', invalid=True)
    valid_count = int(valid.sum())
    return {
        'valid_mask': pd.Series(valid, index=df.index.copy(), name='Valid', dtype=bool),
        'issues': pd.DataFrame(issues, columns=['Source_Row', 'Column', 'Issue']),
        'summary': {'total_rows': count, 'valid_rows': valid_count, 'invalid_rows': count - valid_count,
                    'missing_values': missing_values, 'duplicate_rows': duplicate_rows},
        'missing_columns': missing_columns,
        'can_analyze': bool(count and valid_count and not missing_columns and df.columns.is_unique),
    }
