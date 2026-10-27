"""Launch the standalone browser report-review workflow."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / "web_review" / "server.py"), run_name="__main__")
