"""Local evidence reports; scene-level matching never crosses a recording generation."""
import json
import re
import uuid
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent / "incidents" / "reports"


def persist(incident, directory=REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    incident.setdefault("report_key", uuid.uuid4().hex)
    path = directory / (incident["report_key"] + ".json")
    incident["report_path"] = str(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(incident, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_reports(directory=REPORT_DIR):
    incidents = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if not all(k in item for k in ("category", "summary", "source", "track_id", "severity", "time", "date", "object_label", "class_name", "acknowledged", "dismissed")):
                continue
            item["id"] = len(incidents) + 1
            incidents.append(item)
        except (ValueError, OSError):
            continue
    return incidents


def matches(incident, result, run_id):
    evidence = result["packet"]["evidence"]
    return (incident.get("context_run") == run_id and incident.get("source") == "Video"
            and incident.get("video_seconds") is not None and bool(evidence)
            and evidence[0]["time"] <= incident["video_seconds"] <= result["timestamp"])


class HypothesisHistory:
    def __init__(self):
        self.items = {}
        self.windows = 0

    def update(self, report, timestamp):
        self.windows += 1
        for item in self.items.values():
            item["latest"] = None
        for task in report["plausible_tasks"]:
            key = task.get("family") or legacy_family(task["task"])
            score = task["probability_percent"]
            item = self.items.setdefault(key, dict(task=key.replace("_", " ").title(), peak=-1, at=timestamp))
            item["latest"] = score
            if score > item["peak"]:
                item.update(peak=score, at=timestamp)

    def text(self):
        lines = [f"SESSION PEAKS · {self.windows} assessed windows",
                 "Highest model estimates observed, not final probabilities."]
        for item in sorted(self.items.values(), key=lambda i: i["peak"], reverse=True)[:5]:
            latest = "not proposed in latest window" if item["latest"] is None else f"latest {item['latest']}%"
            lines.append(f"• {item['task']}: peak {item['peak']}% at {item['at']:.1f}s; {latest}")
        lines.append("Hypotheses grouped by task family. Peaks can overstate weak early guesses.")
        return "\n".join(lines)


def legacy_family(name):
    name = name.lower()
    for family, terms in (("maintenance", ("maintenance", "inspection", "repair")),
                          ("tampering", ("unauthorized", "forced entry", "tamper", "lock pick")),
                          ("ordinary_access", ("accessing", "opening", "entryway")),
                          ("waiting", ("waiting", "resting", "loiter", "observation")),
                          ("physical_activity", ("exercise", "exertion", "balance"))):
        if any(term in name for term in terms):
            return family
    return re.sub(r"[^a-z0-9]+", " ", name).strip()
