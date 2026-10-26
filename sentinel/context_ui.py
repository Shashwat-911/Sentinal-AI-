"""Small Tkinter context page sharing the existing single-source video pipeline."""
import uuid
from pathlib import Path
import json
from sentinel.incident_reports import HypothesisHistory, matches, persist, REPORT_DIR

import queue
import time
import tkinter as tk
from tkinter import ttk

from sentinel.context import ContextEngine, MODEL, MODELS
from sentinel.context_report import format_report, format_brief, parse_recording_start, parse_hours


class ContextUI:
    def _build_video_context(self, parent):
        self.context_engine = ContextEngine()
        self.context_generation = 0
        self.context_run = uuid.uuid4().hex
        self.context_history = HypothesisHistory()
        self.context_results = []
        self.context_full_text = "Waiting for evidence."
        self.context_session_reports = []
        self.context_session_errors = []
        self.context_failed = False
        self.context_finished_stamp = None
        self.context_settings = {"enabled": True, "scene": "Unknown (infer visible surroundings)",
                                 "recording_start": "", "expected": "Unknown", "usual_hours": "", "model": MODEL}
        self.context_enabled = tk.BooleanVar(value=True)
        self.context_scene = tk.StringVar(value="Unknown (infer visible surroundings)")
        self.context_start = tk.StringVar()
        self.context_expected = tk.StringVar(value="Unknown")
        self.context_hours = tk.StringVar()
        self.context_model = tk.StringVar(value=MODEL)
        self.context_report_window = None
        self.context_report_text = None
        panel = tk.LabelFrame(parent, text="VIDEO CONTEXT Â· LOCAL VISION", bg="#151b23",
                              fg="#5fd38d", font=("Segoe UI", 11, "bold"), padx=8, pady=5)
        panel.pack(fill="both", expand=True, padx=10, pady=5)
        controls = tk.Frame(panel, bg="#151b23")
        controls.pack(fill="x")
        ttk.Checkbutton(controls, text="Smart context", variable=self.context_enabled,
                        command=self._apply_context).pack(side="left")
        ttk.Button(controls, text="Apply", command=self._apply_context).pack(side="right")
        ttk.Button(controls, text="Full report", command=self._open_context_report).pack(side="right", padx=2)
        ttk.Button(controls, text="Environment", command=self._context_options).pack(side="right", padx=2)
        form = tk.Frame(panel, bg="#151b23")
        form.pack(fill="x", pady=3)
        form.columnconfigure(1, weight=1)
        for row, (label, var) in enumerate([("Scene (optional)", self.context_scene),
                                           ("Recorded at (optional)", self.context_start)]):
            tk.Label(form, text=label, bg="#151b23", fg="#8b98a8", font=("Segoe UI", 9)).grid(row=row,column=0,sticky="w")
            ttk.Entry(form, textvariable=var, width=22).grid(row=row,column=1,sticky="ew",padx=4,pady=2)
        self.context_status = tk.StringVar(value="Load a video. Frames + pose + movement will be analysed together.")
        tk.Label(panel, textvariable=self.context_status, wraplength=390, anchor="w", justify="left",
                 bg="#151b23", fg="#8b98a8", font=("Segoe UI", 9)).pack(fill="x", pady=4)
        text_frame = tk.Frame(panel, bg="#151b23")
        text_frame.pack(fill="both", expand=True)
        self.context_text = tk.Text(text_frame, wrap="word", width=40, height=10, bg="#1b2430",
                                    fg="#e6edf3", font=("Segoe UI", 10), relief="flat", padx=8, pady=8)
        scroll = ttk.Scrollbar(text_frame, command=self.context_text.yview)
        scroll.pack(side="right", fill="y")
        self.context_text.configure(yscrollcommand=scroll.set)
        self.context_text.pack(fill="both", expand=True)
        self._set_text(self.context_text, "The original detector, skeletons, behavior scores and alerts remain active.\n\nVisual context reviews a short sequence, without waiting for a dwell threshold. It may be uncertain or mistaken.")
        self.root.after(100, self._poll_context)

    def _context_options(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Recording environment and local model")
        dialog.configure(bg="#151b23")
        dialog.transient(self.root)
        form = ttk.Frame(dialog, padding=16)
        form.pack(fill="both", expand=True)
        fields = [("Scene / location type", self.context_scene),
                  ("Recording date and time", self.context_start),
                  ("Usual activity hours", self.context_hours)]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(form, textvariable=variable, width=42).grid(row=row, column=1, padx=10, pady=5)
        ttk.Label(form, text="Activity expected?").grid(row=3, column=0, sticky="w")
        ttk.Combobox(form, textvariable=self.context_expected, values=["Unknown", "Yes", "No"],
                     state="readonly").grid(row=3, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(form, text="Local vision model").grid(row=4, column=0, sticky="w")
        ttk.Combobox(form, textvariable=self.context_model, values=list(MODELS),
                     state="readonly", width=28).grid(row=4, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(form, text="Date/time: 2026-10-26T14:00:00+05:30. Blank = unknown.\n"
                            "Hours: 09:00-18:00 (also supports overnight hours).\n"
                            "These are operator-supplied facts, not facts inferred from the video.\n"
                            "4b: more detailed model; 2b: smaller, faster option.",
                  wraplength=520).grid(row=5,column=0,columnspan=2,sticky="w",pady=12)
        def save():
            if self._apply_context():
                dialog.destroy()
        ttk.Button(form, text="Apply environment", command=save).grid(row=6,column=1,sticky="e")

    def _open_context_report(self):
        if self.context_report_window and self.context_report_window.winfo_exists():
            self.context_report_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.context_report_window = window
        window.title("Detailed behavioral assessment â€” model interpretation")
        window.geometry("850x700")
        window.transient(self.root)
        text = tk.Text(window, wrap="word", bg="#151b23", fg="#e6edf3", font=("Segoe UI", 11), padx=18,pady=18)
        bar = ttk.Scrollbar(window, command=text.yview)
        bar.pack(side="right",fill="y")
        text.configure(yscrollcommand=bar.set)
        text.pack(fill="both",expand=True)
        self.context_report_text = text
        self._sync_context_report()

    def _sync_context_report(self):
        if self.context_report_window and self.context_report_window.winfo_exists():
            self._set_text(self.context_report_text, self.context_full_text)

    def _apply_context(self):
        start = self.context_start.get().strip()
        try:
            if start:
                parse_recording_start(start)
            parse_hours(self.context_hours.get())
        except ValueError as exc:
            self.context_status.set(str(exc) + ". Date/time example: 2026-10-26T14:00:00+05:30")
            return False
        self.context_settings = {"enabled": self.context_enabled.get(),
                                 "scene": self.context_scene.get().strip()[:200] or "Unknown",
                                 "recording_start": start, "expected": self.context_expected.get(),
                                 "usual_hours": self.context_hours.get().strip(), "model": self.context_model.get()}
        self._reset_context()
        return True

    def _reset_context(self):
        self.context_run = uuid.uuid4().hex
        self.context_history = HypothesisHistory()
        self.context_results = []
        self.context_full_text = "Waiting for new evidence. Previous assessment cleared."
        self.context_session_reports = []
        self.context_session_errors = []
        self.context_failed = False
        self.context_finished_stamp = None
        self.context_generation = self.context_engine.reset()
        self.context_status.set("Collecting fresh evidence..." if self.context_settings["enabled"] else "Context analysis disabled")
        self._set_text(self.context_text, "Waiting for new evidence. Previous assessment cleared.")
        self._sync_context_report()

    def _observe_context(self, generation, source, timestamp, tracks, brightness, frame=None):
        # Called by the inference worker; never read or mutate Tk widgets here.
        self.context_engine.observe(generation, source, timestamp, tracks, brightness, self.context_settings, frame=frame)

    def _poll_context(self):
        try:
            while True:
                result = self.context_engine.results.get_nowait()
                if result["generation"] != self.context_generation:
                    continue
                packet = result["packet"]
                label = f"Evidence through {result['timestamp']:.1f}s" if packet["source"] == "vid" else f"Captured {packet['capture_time']}"
                self.context_status.set(f"{label} | Local response: {result['latency']}s | {packet.get('model', MODEL)}")
                if "error" in result:
                    for incident in self.incidents:
                        if matches(incident, result, self.context_run):
                            incident["context_error"] = result["error"]
                            self._save_incident_report(incident)
                    self.context_session_errors.append({"timestamp": result["timestamp"], "error": result["error"]})
                    self.context_failed = True
                    self.context_full_text = result["error"]
                    self._save_session_report()
                    self._set_text(self.context_text, result["error"] + "\n\nDetection continues. Start Ollama and ensure the selected model is installed (Environment settings).")
                    self._sync_context_report()
                    continue
                self.context_failed = False
                assessment = result["assessment"]
                self.context_history.update(assessment, result["timestamp"])
                self.context_session_reports.append({"timestamp": result["timestamp"], "assessment": assessment})
                self.context_results.append(result)
                self.context_results = self.context_results[-12:]
                text = format_report(assessment, packet, result["timestamp"])
                text += "\n\nMEASURED TELEMETRY (ORIGINAL PIPELINE)\n" + "\n".join(
                    f"Person {p['track_id']}: {p['posture']}; dwell {p['dwell_seconds']}s; "
                    f"revisits {p['revisits']}; speed {p.get('speed_px_per_second', 0)}px/s; {p['behavior']}"
                    for p in packet["evidence"][-1]["people"])
                if not packet["evidence"][-1]["people"]:
                    text += "No tracked person in the latest detector sample; visual review uses the sampled images."
                self.context_full_text = text + "\n\n" + self.context_history.text()
                self._set_text(self.context_text, format_brief(assessment, packet, result["timestamp"]))
                self._sync_context_report()
                self._attach_context_to_incidents(result)
                self._save_session_report()
        except queue.Empty:
            pass
        busy = self.context_engine.busy
        if (not getattr(self, "video_running", False) and not busy
                and not self.context_engine.has_pending() and self.context_results and not self.context_failed
                and self.context_finished_stamp != self.context_results[-1]["timestamp"]):
            latest = self.context_results[-1]
            self.context_finished_stamp = latest["timestamp"]
            self._set_text(self.context_text, format_brief(latest["assessment"], latest["packet"], latest["timestamp"])
                           + "\n\n" + self.context_history.text())
        if busy and busy[0] == self.context_generation:
            self.context_status.set(f"Local model analyzing ({time.monotonic() - busy[1]:.0f}s). Playback and analysis run independently.")
        self.root.after(100, self._poll_context)

    def _save_session_report(self):
        try:
            folder = REPORT_DIR / "sessions"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (self.context_run + ".txt")
            payload = {"video": getattr(self, "video_path", None), "reports": self.context_session_reports,
                       "errors": self.context_session_errors, "peaks": self.context_history.items}
            for target, contents in ((path.with_suffix(".json"), json.dumps(payload, indent=2)),
                                     (path, "Video: " + str(getattr(self, "video_path", "unknown")) + "\n\n" + self.context_full_text)):
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_text(contents, encoding="utf-8")
                temporary.replace(target)
        except OSError as exc:
            self.context_status.set(f"Report could not be saved: {exc}")

    def _save_incident_report(self, incident):
        try:
            persist(incident)
            incident.pop("report_error", None)
        except OSError as exc:
            incident["report_error"] = str(exc)

    def _attach_context_to_incidents(self, result):
        changed = False
        for incident in self.incidents:
            if matches(incident, result, self.context_run):
                entry = {key: value for key, value in result.items() if key != "packet"}
                entry["packet"] = {key: value for key, value in result["packet"].items() if key != "images"}
                history = incident.setdefault("context_reports", [])
                if any(item["timestamp"] == result["timestamp"] for item in history):
                    continue
                incident.pop("context_error", None)
                history.append(entry)
                incident["context_brief"] = format_brief(result["assessment"], result["packet"], result["timestamp"])
                self._save_incident_report(incident)
                changed = True
        if changed:
            self._refresh_incidents_view()
