"""Run the test suite and save a clean UTF-8 report and machine-readable summary."""

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import io
import json
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))
output = io.StringIO()
with redirect_stdout(output), redirect_stderr(output):
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
report = output.getvalue()
(root / 'results' / 'test_report.txt').write_text(report, encoding='utf-8')
summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'tests_run': result.testsRun,
           'failures': len(result.failures), 'errors': len(result.errors),
           'skipped': len(result.skipped), 'passed': result.wasSuccessful()}
(root / 'results' / 'test_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(report)
sys.exit(0 if result.wasSuccessful() else 1)
