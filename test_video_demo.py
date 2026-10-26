"""Video startup/error regression tests using a small local synthetic clip."""
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import app as dashboard
from test_context import reply


class VideoDemoTests(unittest.TestCase):
    def test_background_loading_preview_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "sample.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 64))
            self.assertTrue(writer.isOpened())
            for index in range(12):
                writer.write(np.full((64, 64, 3), index * 10, dtype=np.uint8))
            writer.release()
            root = tk.Tk()
            root.withdraw()
            ui = dashboard.SentinelApp(root)
            ui.context_engine.reasoner = lambda packet: reply()
            ui._save_session_report = lambda: None
            released, entered = threading.Event(), threading.Event()
            def load(_):
                entered.set()
                released.wait(3)
                return object(), "test model"
            def pump_until(predicate, timeout=3):
                deadline = time.monotonic() + timeout
                while not predicate() and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.01)
                self.assertTrue(predicate())
            try:
                self.assertNotIn("context", ui.app_pages)
                self.assertTrue(ui._select_video(path))
                self.assertIsNotNone(ui._last_frame)
                ui.context_start.set("2026-10-26T23:00:00+05:30")
                ui.context_hours.set("09:00-18:00")
                self.assertTrue(ui._apply_context())
                self.assertEqual(ui.context_settings["usual_hours"], "09:00-18:00")
                ui.context_engine.results.put(dict(
                    generation=ui.context_generation, timestamp=1, latency=1,
                    assessment=reply(), packet=dict(source="vid", capture_time="unknown",
                    evidence=[dict(id="E1", people=[])])))
                ui._poll_context()
                ui._open_context_report()
                self.assertIn("POSSIBLE TASKS", ui.context_report_text.get("1.0", "end"))
                ui._reset_context()
                self.assertNotIn("Waiting (possible)", ui.context_report_text.get("1.0", "end"))
                ui.context_report_window.destroy()
                ui._save_incident_report = lambda item: None
                incident = dict(id=9999, source="Video", category="REPEATED_MOVEMENT", severity="HIGH",
                    track_id=1, class_name="person", object_label="PERSON#1", summary="Original trigger",
                    time="12:00", date="2026-10-26", acknowledged=False, dismissed=False,
                    context_run=ui.context_run, video_seconds=.5)
                ui.incidents.append(incident)
                ui._refresh_incidents_view()
                ui.incident_tree.selection_set("9999")
                ui._open_incident_report()
                before = ui.incident_report_text.get("1.0", "end")
                ui._attach_context_to_incidents(dict(timestamp=1, assessment=reply(),
                    packet=dict(capture_time="unknown", evidence=[dict(time=0, id="E1")]), generation=ui.context_generation))
                pump_until(lambda: ui.incident_report_text.get("1.0", "end") != before)
                self.assertIn("LATEST MATCHING SCENE REVIEW", ui.incident_report_text.get("1.0", "end"))
                ui.incident_report_window.destroy()
                ui.incidents.remove(incident)

                with patch.object(dashboard, "load_person_model", side_effect=load), \
                     patch.object(dashboard, "load_pose_model", return_value=None), \
                     patch.object(dashboard, "detect_persons", return_value=[]), \
                     patch.object(dashboard, "estimate_postures", return_value=[]), \
                     patch.object(dashboard, "draw_overlays", side_effect=lambda frame, *a, **kw: (frame, [])):
                    started = time.monotonic()
                    ui._play_video()
                    self.assertLess(time.monotonic() - started, .5)
                    self.assertTrue(entered.wait(1))
                    pulses = []
                    root.after(10, lambda: pulses.append(True))
                    pump_until(lambda: bool(pulses))
                    released.set()
                    pump_until(lambda: not ui.video_running)
                    self.assertEqual(ui.video_source_status, "Finished")
                    self.assertEqual(ui.vid_fcount, 12)
                    ui._play_video()
                    pump_until(lambda: not ui.video_running)
                    self.assertEqual(ui.vid_fcount, 12)
                self.assertFalse(ui._select_video(str(Path(folder) / "missing.mp4")))
                self.assertEqual(str(ui.btn_vid_play["state"]), "disabled")
                self.assertTrue(ui._select_video(path))
            finally:
                released.set()
                ui.video_running = False
                if ui.source_thread:
                    ui.source_thread.join(timeout=3)
                ui._on_close()


if __name__ == "__main__":
    unittest.main()
