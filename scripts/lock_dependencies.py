"""Record exact runtime dependency versions for this project's verified environment."""

from importlib import metadata
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
pending = [Requirement(line).name for line in (root / 'requirements.txt').read_text().splitlines()
           if line.strip() and not line.startswith('#')]
resolved = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in resolved:
        continue
    distribution = metadata.distribution(name)
    resolved[name] = distribution.version
    for dependency in distribution.requires or []:
        requirement = Requirement(dependency)
        if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
            pending.append(requirement.name)

lines = ['# Verified Python 3.12 environment. Regenerate with scripts/lock_dependencies.py.']
lines.extend(f'{name}=={version}' for name, version in sorted(resolved.items()))
(root / 'requirements-lock.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Locked {len(resolved)} runtime dependencies.')
