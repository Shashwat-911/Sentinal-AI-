import tempfile
import unittest
from pathlib import Path
from sentinel.incident_reports import persist, load_reports, matches, HypothesisHistory
from sentinel.context_report import validate_report, format_brief
from test_context import reply

class ReportTests(unittest.TestCase):
    def test_matching_requires_same_run_and_covering_window(self):
        incident = dict(context_run="one", source="Video", video_seconds=10)
        result = dict(timestamp=14, packet=dict(evidence=[dict(time=2)]))
        self.assertTrue(matches(incident, result, "one"))
        self.assertFalse(matches(incident, result, "two"))
        incident["video_seconds"] = 15
        self.assertFalse(matches(incident, result, "one"))
        incident["video_seconds"] = 1
        self.assertFalse(matches(incident, result, "one"))

    def test_peak_retains_timestamp_and_lower_latest(self):
        history = HypothesisHistory()
        report = reply()
        history.update(report, 5)
        report["plausible_tasks"][0]["probability_percent"] = 20
        history.update(report, 10)
        self.assertIn("peak 50% at 5.0s; latest 20%", history.text())
        report["plausible_tasks"] = []
        history.update(report, 15)
        self.assertIn("not proposed in latest window", history.text())

    def test_estimates_validated_and_brief_bounded(self):
        report = reply()
        packet = dict(evidence=[dict(id="E1")])
        for invalid in (-1, 101, True, "50", 4.2):
            report["plausible_tasks"][0]["probability_percent"] = invalid
            with self.assertRaises(ValueError):
                validate_report(report, packet)
        report = reply()
        report["behavior_summary"] = "x" * 1700
        self.assertLess(len(format_brief(report, packet, 10)), 850)
        self.assertIn("50%", format_brief(report, packet, 10))

    def test_context_attaches_without_images_and_keeps_trigger(self):
        from sentinel.context_ui import ContextUI
        ui = ContextUI()
        ui.context_run = "run"
        ui.incidents = [dict(source="Video", context_run="run", video_seconds=5, summary="Original trigger")]
        saved = []
        ui._save_incident_report = lambda item: saved.append(item)
        ui._refresh_incidents_view = lambda: None
        result = dict(timestamp=10, generation=1, assessment=reply(),
                      packet=dict(evidence=[dict(time=2, id="E1")], images=["privatepixels"]))
        ui._attach_context_to_incidents(result)
        ui._attach_context_to_incidents(result)
        self.assertEqual(len(saved), 1)
        self.assertEqual(ui.incidents[0]["summary"], "Original trigger")
        self.assertNotIn("images", ui.incidents[0]["context_reports"][0]["packet"])

    def test_legacy_synonyms_merge_into_one_family(self):
        history = HypothesisHistory()
        report = reply()
        report["plausible_tasks"] = [dict(task="Routine maintenance or inspection", probability_percent=45)]
        history.update(report, 2)
        report["plausible_tasks"] = [dict(task="Routine inspection or maintenance", probability_percent=35)]
        history.update(report, 10)
        self.assertEqual(len(history.items), 1)
        self.assertEqual(history.items["maintenance"]["latest"], 35)

    def test_brightness_does_not_become_recording_time(self):
        from sentinel.context_report import lighting_text
        packet = dict(evidence=[dict(brightness_0_255=20)], capture_time="unknown")
        text = lighting_text(reply(), packet)
        self.assertIn("dim image", text)
        self.assertIn("indeterminate", text)
        self.assertEqual(packet["capture_time"], "unknown")

    def test_output_review_does_not_change_verdict(self):
        from sentinel.context_report import review_output
        report = reply()
        report["lighting_assessment"] = dict(appearance="artificial_or_night_like", reason="Low brightness 110")
        report["missing_information"] = ["Identity of the person", "Whether the door opens"]
        report["reassuring_factors"] = ["No other individuals present", "Person leaves the object undamaged"]
        reviewed = review_output(report)
        self.assertEqual(reviewed["classification"], "uncertain")
        self.assertEqual(reviewed["lighting_assessment"]["appearance"], "indeterminate")
        self.assertEqual(reviewed["missing_information"], ["Whether the door opens"])
        self.assertEqual(len(reviewed["reassuring_factors"]), 1)

    def test_saved_report_survives_reload_and_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            incident = dict(category="EVENT", summary="trigger", source="Video", track_id=1,
                severity="HIGH", time="12:00", date="2026-10-26", object_label="PERSON#1",
                class_name="person", acknowledged=False, dismissed=False)
            persist(incident, folder)
            incident["context_reports"] = [dict(assessment=reply())]
            persist(incident, folder)
            loaded = load_reports(folder)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["context_reports"][0]["assessment"], reply())

if __name__ == "__main__":
    unittest.main()
