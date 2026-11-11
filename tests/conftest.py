import sys
from pathlib import Path

# Lets tests import helpers that live next to them (synthetic_statcast).
sys.path.insert(0, str(Path(__file__).parent))
