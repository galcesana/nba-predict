"""Repository-root Streamlit entry point for local and cloud deployment."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.app.streamlit_app import main  # noqa: E402

main()
