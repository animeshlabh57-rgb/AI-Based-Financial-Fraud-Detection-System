"""Create a portable submission archive without environments, caches or backups."""

from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
files = ['README.md', 'main.py', 'config.py', 'requirements.txt', 'requirements-lock.txt',
         'START_PROJECT.bat', 'run_dashboard.ps1', '.gitignore']
folders = ['src', 'dashboard', 'data', 'models', 'results', 'tests', 'scripts', 'logs', 'docs']
paths = [root / filename for filename in files]
for folder in folders:
    paths.extend(path for path in (root / folder).rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
# Account data stays local even if packaging folders are expanded later.
assert all('storage' not in path.relative_to(root).parts and
           path.suffix.lower() not in {'.sqlite', '.sqlite3', '.db'} and
           not path.name.endswith(('-wal', '-shm')) for path in paths), 'Private workspace storage must not be packaged.'
archive_path = root / 'FraudGuard_AI_Submission.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths):
        archive.write(path, 'FraudGuard_AI/' + path.relative_to(root).as_posix())
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None, 'Archive integrity check failed.'
    print(f'{len(archive.namelist())} files packaged.')
print(f'{archive_path} ({archive_path.stat().st_size / 1024 / 1024:.2f} MB)')
