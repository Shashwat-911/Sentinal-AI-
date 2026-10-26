"""Context contract, timing and worker lifecycle regression tests (no model needed)."""
import queue
import threading
import unittest

from sentinel.context import ContextEngine, capture_time, validate
from sentinel.context_report import temporal_context, parse_hours, format_report


def reply():
    return dict(classification="uncertain", evidence_strength="limited", behavior_summary="Standing",
                action_sequence=["Person stands"], scene_analysis=["Surface partly visible"],
                context_effect="Recording time unknown", lighting_assessment=dict(appearance="indeterminate", reason="No natural lighting cues"), plausible_tasks=[
                    dict(family="waiting", task="Waiting", probability_percent=50, plausibility="possible", support="Standing", limitations="Purpose unknown"),
                    dict(family="physical_activity", task="Resting", probability_percent=40, plausibility="possible", support="Stationary", limitations="Purpose unknown")],
                concern_factors=[], reassuring_factors=[], explanation="Insufficient action evidence",
                missing_information=["Purpose"], evidence_ids=["E1"])



class ContextTests(unittest.TestCase):
    def test_date_time_and_overnight_hours(self):
        context = temporal_context("2026-10-26T02:30:00+05:30", "22:00-06:00")
        self.assertEqual(context["weekday"], "Monday")
        self.assertEqual(context["clock_period"], "overnight")
        self.assertEqual(context["usual_hours_status"], "within supplied usual hours")
        context = temporal_context("2026-10-26T06:00:00+05:30", "22:00-06:00")
        self.assertEqual(context["usual_hours_status"], "outside supplied usual hours")
        self.assertEqual(temporal_context("unknown", "09:00-18:00")["weekday"], "unknown")
        with self.assertRaises(ValueError):
            capture_time("vid", 10, "2026-10-26")
        for invalid in ("25:00-30:00", "09:00-09:00", "9am-5pm"):
            with self.assertRaises(ValueError):
                parse_hours(invalid)

    def test_hypotheses_are_validated_and_displayed_as_possibilities(self):
        report = reply()
        packet = dict(evidence=[dict(id="E1")], capture_time="unknown")
        validate(report, packet)
        rendered = format_report(report, packet, 10)
        self.assertIn("POSSIBLE TASKS", rendered)
        self.assertIn("Waiting (possible)", rendered)
        self.assertIn("REASONS FOR CONCERN", rendered)
        report["plausible_tasks"][0]["plausibility"] = "99%"
        with self.assertRaises(ValueError):
            validate(report, packet)

    def test_visual_context_does_not_wait_for_dwell_or_detection(self):
        import numpy as np
        engine = ContextEngine(lambda packet: reply())
        settings = dict(enabled=True, scene="Unknown", recording_start="", expected="Unknown")
        try:
            for timestamp in (0, 1, 2):
                engine.observe(0, "vid", timestamp, {}, 50, settings,
                               frame=np.zeros((100, 100, 3), dtype=np.uint8))
            result = engine.results.get(timeout=2)
            packet = result["packet"]
            self.assertEqual(len(packet["images"]), 3)
            self.assertEqual([f["video_seconds"] for f in packet["visual_frames"]], [0, 1, 2])
            self.assertEqual(packet["evidence"][-1]["people"], [])
            self.assertEqual(packet["rule_assessment"], "observe")
            engine.reset()
            self.assertFalse(engine.frames)
        finally:
            engine.close()

    def test_final_short_window_is_submitted_without_duplicate_evidence(self):
        engine = ContextEngine(lambda packet: reply())
        settings = dict(enabled=True, scene="Unknown", recording_start="", expected="Unknown")
        try:
            engine.observe(0, "vid", .5, {}, 50, settings)
            engine.observe(0, "vid", .5, {}, 50, settings, force=True)
            result = engine.results.get(timeout=2)
            self.assertEqual(result["timestamp"], .5)
            self.assertEqual(len(result["packet"]["evidence"]), 1)
        finally:
            engine.close()

    def test_priority_incident_survives_routine_replacement(self):
        entered, release = threading.Event(), threading.Event()
        def reasoner(packet):
            entered.set()
            release.wait(2)
            return reply()
        engine = ContextEngine(reasoner)
        settings = dict(enabled=True, scene="Unknown", recording_start="", expected="Unknown")
        try:
            engine.observe(0, "vid", 1, {}, 50, settings, force=True)
            self.assertTrue(entered.wait(1))
            engine.observe(0, "vid", 2, {}, 50, settings, force=True, urgent=True)
            engine.observe(0, "vid", 3, {}, 50, settings, force=True)
            engine.observe(0, "vid", 4, {}, 50, settings, force=True)
            release.set()
            times = [engine.results.get(timeout=3)["timestamp"] for _ in range(3)]
            self.assertEqual(times, [1, 2, 4])
        finally:
            release.set()
            engine.close()

    def test_rule_review_survives_latest_detector_miss(self):
        engine = ContextEngine(lambda packet: reply())
        settings = dict(enabled=True, scene="Unknown", recording_start="", expected="Unknown")
        try:
            engine.observe(0, "vid", 0, {1: dict(track_id=1, active_threats=["REPEATED_MOVEMENT"])}, 100, settings)
            engine.observe(0, "vid", 1, {}, 100, settings)
            engine.observe(0, "vid", 2, {}, 100, settings)
            result = engine.results.get(timeout=2)
            self.assertEqual(result["packet"]["rule_assessment"], "review")
            self.assertEqual(result["packet"]["evidence"][-1]["people"], [])
        finally:
            engine.close()

    def test_old_routine_window_does_not_overwrite_newer_incident_review(self):
        entered, release = threading.Event(), threading.Event()
        def reasoner(packet):
            entered.set()
            release.wait(2)
            return reply()
        engine = ContextEngine(reasoner)
        settings = dict(enabled=True, scene="Unknown", recording_start="", expected="Unknown")
        try:
            engine.observe(0, "vid", 1, {}, 50, settings, force=True)
            self.assertTrue(entered.wait(1))
            engine.observe(0, "vid", 2, {}, 50, settings, force=True)
            engine.observe(0, "vid", 3, {}, 50, settings, force=True, urgent=True)
            release.set()
            self.assertEqual([engine.results.get(timeout=3)["timestamp"] for _ in range(2)], [1, 3])
            with self.assertRaises(queue.Empty):
                engine.results.get(timeout=.3)
        finally:
            release.set()
            engine.close()

    def test_disabled_context_sends_no_frames(self):
        engine = ContextEngine(lambda packet: reply())
        try:
            for timestamp in range(10):
                engine.observe(0, "vid", timestamp, {}, 50,
                               dict(enabled=False, scene="", recording_start="", expected="Unknown"))
            self.assertFalse(engine.history)
            self.assertTrue(engine.pending.empty())
        finally:
            engine.close()

    def test_recorded_time_never_uses_system_clock(self):
        self.assertEqual(capture_time("vid", 10, ""), "unknown")
        self.assertEqual(capture_time("vid", 90, "2026-10-25T23:59:00+05:30"),
                         "2026-10-26T00:00:30+05:30")

    def test_invented_evidence_is_rejected(self):
        result = reply()
        validate(result, {"evidence": [{"id": "E1"}]})
        with self.assertRaises(ValueError):
            validate(result, {"evidence": [{"id": "E2"}]})
        result["classification"] = "criminal"
        with self.assertRaises(ValueError):
            validate(result, {"evidence": [{"id": "E1"}]})

    def test_reset_discards_inflight_reply(self):
        entered, release = threading.Event(), threading.Event()
        def reasoner(packet):
            entered.set()
            release.wait(2)
            return reply()
        engine = ContextEngine(reasoner)
        try:
            engine.pending.put((0, 5, {}))
            self.assertTrue(entered.wait(1))
            engine.reset()
            release.set()
            with self.assertRaises(queue.Empty):
                engine.results.get(timeout=.3)
        finally:
            release.set()
            engine.close()

    def test_history_and_queue_are_bounded_and_unknown_time_preserved(self):
        entered, release = threading.Event(), threading.Event()
        def reasoner(packet):
            entered.set()
            release.wait(2)
            return reply()
        engine = ContextEngine(reasoner)
        settings = dict(enabled=True, scene="Hall", recording_start="", expected="Unknown")
        try:
            for timestamp in range(100):
                engine.observe(0, "vid", timestamp, {1: dict(track_id=1, dwell_time=timestamp)}, 30, settings)
            self.assertLessEqual(len(engine.history), 30)
            self.assertLessEqual(engine.pending.qsize(), 1)
            self.assertTrue(entered.wait(1))
            release.set()
            result = engine.results.get(timeout=1)
            self.assertEqual(result["packet"]["capture_time"], "unknown")
            self.assertEqual(result["packet"]["rule_assessment"], "observe")
        finally:
            release.set()
            engine.close()

    def test_failure_does_not_kill_worker(self):
        def fail(packet):
            raise TimeoutError("test")
        engine = ContextEngine(fail)
        try:
            engine.pending.put((0, 1, {}))
            self.assertIn("TimeoutError", engine.results.get(timeout=1)["error"])
            self.assertTrue(engine.worker.is_alive())
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
