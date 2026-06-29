import sys
from pathlib import Path

# Force local package import priority so pytest validates repository code,
# not a globally installed SemanticDocumentParser version.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
