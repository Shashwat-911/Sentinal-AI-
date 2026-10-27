"""Validate incident exports using the desktop application's report persistence."""
import importlib.util
import tempfile
from pathlib import Path

# Load the stdlib-only report module without initializing the camera/pose package.
path = Path(__file__).resolve().parents[1] / "sentinel" / "incident_reports.py"
spec = importlib.util.spec_from_file_location("sentinel_report_storage", path)
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)
TEXT_FIELDS = ("category", "summary", "source", "severity", "time", "date", "object_label", "class_name")

def preview(payload):
    rows = payload if isinstance(payload, list) else [payload]
    if not 1 <= len(rows) <= 100:
        raise ValueError("Provide between 1 and 100 incident reports.")
    validated = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each incident must be an object.")
        item = dict(row)
        for field in TEXT_FIELDS:
            if not isinstance(item.get(field), str) or len(item[field]) > 20000:
                raise ValueError(f"Missing or invalid field: {field}")
        if isinstance(item.get("track_id"), bool) or not isinstance(item.get("track_id"), (int, str)):
            raise ValueError("track_id must be an integer or string.")
        for field in ("acknowledged", "dismissed"):
            if not isinstance(item.get(field), bool):
                raise ValueError(f"{field} must be a boolean.")
        if item["severity"] not in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "NORMAL"):
            raise ValueError("Unknown severity.")
        # Imported report keys/paths must never choose a filesystem destination.
        item.pop("report_key", None)
        item.pop("report_path", None)
        validated.append(item)
    with tempfile.TemporaryDirectory(prefix="sentinel-review-") as folder:
        directory = Path(folder)
        for item in validated:
            storage.persist(item, directory)
        reports = storage.load_reports(directory)
        for row in reports:
            row.pop("report_path", None)
    return {"reports": reports}
