"""Entry point: ``streamlit run streamlit_app.py``."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from news_radar.app import main  # noqa: E402

main()
