"""Make the repo root importable so `import src` / `import api` work under pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
