"""Context contract, timing and worker lifecycle regression tests (no model needed)."""
import queue
import threading
import unittest

from sentinel.context import ContextEngine, capture_time, validate
from sentinel.context_report import temporal_context, parse_hours, format_report


def reply():
    return dict(subject_scope="people", classification="uncertain", evidence_strength="limited", behavior_summary="Standing",
                action_sequence=["Person stands"], observed_events=[], scene_analysis=["Surface partly visible"],
                context_effect="Recording time unknown", lighting_assessment=dict(appearance="indeterminate", reason="No natural lighting cues"), plausible_tasks=[
                    dict(family="waiting", task="Waiting", probability_percent=50, plausibility="possible", support="Standing", limitations="Purpose unknown"),
                    dict(family="physical_activity", task="Resting", probability_percent=40, plausibility="possible", support="Stationary", limitations="Purpose unknown")],
                concern_factors=[], reassuring_factors=[], explanation="Insufficient action evidence",
                missing_information=["Purpose"], evidence_ids=["E1"])



class ContextTests(unittest.TestCase):
    def test_person_detail_keeps_nearby_context_and_bounds(self):
        import numpy as np
        from sentinel.context_views import person_detail
        frame=np.zeros((240,320,3),dtype=np.uint8)
        self.assertIsNone(person_detail(frame,[]))
        self.assertIsNone(person_detail(frame,[(0,0,320,240)]))
        self.assertIsNone(person_detail(frame,[(float('nan'),0,10,10)]))
        crop,rect=person_detail(frame,[(60,80,140,205)])
        self.assertLess(rect[0],60)
        self.assertGreater(rect[2],140)
        self.assertLess(rect[1],80)
        self.assertGreaterEqual(rect[3],205)
        self.assertEqual(crop.shape[:2],(rect[3]-rect[1],rect[2]-rect[0]))

    def test_global_view_changes_do_not_count_as_subject_motion(self):
        import numpy as np
        from sentinel.context import frame_change
        before = np.zeros((120,160), dtype=np.uint8)
        local = before.copy()
        local[20:50,30:50] = 255
        self.assertFalse(frame_change(before, local)[1])
        self.assertTrue(frame_change(before, np.full_like(before, 100))[1])

    def test_view_change_excludes_earlier_context_evidence(self):
        import numpy as np
        engine = ContextEngine(lambda packet: reply())
        settings = dict(enabled=True, scene='Unknown', recording_start='', expected='Unknown')
        try:
            for t in range(8):
                frame = np.full((40,40,3), 180 if t >= 4 else 0, dtype=np.uint8)
                engine.observe(0, 'vid', t, {}, 100, settings, frame=frame)
            engine.observe(0, 'vid', 7, {}, 100, settings, frame=frame, force=True)
            while True:
                result = engine.results.get(timeout=3)
                if result['timestamp'] == 7: break
            self.assertEqual(result['packet']['scene_boundary_seconds'], 4)
            self.assertTrue(all(e['time'] >= 4 for e in result['packet']['evidence']))
            self.assertTrue(all(f['video_seconds'] >= 4 for f in result['packet']['visual_frames']))
            self.assertIn('change detected at 4.0s', format_report(reply(), result['packet'], 7))
            engine.reset()
            self.assertIsNone(engine.scene_boundary)
        finally:
            engine.close()

    def test_camera_translation_preserves_shared_scene_features(self):
        import cv2
        import numpy as np
        from sentinel.context import frame_change
        before = np.random.default_rng(7).integers(0, 256, (120,160), dtype=np.uint8)
        shifted = cv2.warpAffine(before, np.float32([[1,0,7],[0,1,4]]), (160,120))
        self.assertFalse(frame_change(before, shifted)[1])
        unrelated = np.random.default_rng(8).integers(0, 256, (120,160), dtype=np.uint8)
        self.assertTrue(frame_change(before, unrelated)[1])

    def test_large_foreground_occlusion_preserves_a_locally_matching_view(self):
        import cv2
        import numpy as np
        from sentinel.context import frame_change
        background = np.full((120,160), 80, dtype=np.uint8)
        cv2.rectangle(background, (105,82), (154,118), 130, -1)
        for x in range(109,153,8):
            cv2.line(background, (x,84), (x-2,116), 220, 2)
        cv2.circle(background, (128,100), 8, 20, 2)
        cv2.line(background, (105,118), (154,84), 250, 2)
        before, after = background.copy(), background.copy()
        cv2.rectangle(before, (0,10), (85,115), 210, -1)
        cv2.rectangle(after, (15,10), (100,115), 20, -1)
        motion, changed = frame_change(before, after)
        self.assertGreater(motion, 25)
        self.assertFalse(changed)

    def test_report_exposes_visual_gaps_not_telemetry_density(self):
        packet = dict(capture_time='unknown', evidence=[dict(time=t) for t in range(30)],
                      visual_frames=[dict(video_seconds=t) for t in (0,12,13,14,29)])
        text = format_report(reply(),packet,29)
        self.assertIn('Visual samples: 5; widest gap 15.0s',text)

    def test_vl_report_uses_observation_record_and_keeps_images_local_to_first_pass(self):
        import base64
        import io
        import json
        from unittest.mock import patch
        import cv2
        import numpy as np
        from sentinel.context import assess
        encoded = base64.b64encode(cv2.imencode('.jpg', np.zeros((24,32,3), dtype=np.uint8))[1]).decode()
        packet = dict(images=[encoded], evidence=[dict(id='E1', time=1)],
                      visual_frames=[dict(evidence_id='E1', video_seconds=1)])
        calls = []
        def respond(request, **kwargs):
            body = json.loads(request.data)
            calls.append(body)
            content = 'E1: A person stands in daylight.' if len(calls) == 1 else json.dumps(reply())
            return io.BytesIO(json.dumps(dict(message=dict(content=content))).encode())
        with patch('sentinel.context.urllib.request.urlopen', respond):
            report = assess(packet, 'qwen3-vl:4b-instruct')
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]['messages'][0]['images']), 1)
        self.assertIn('evidence E1: [img]', calls[0]['messages'][0]['content'])
        self.assertNotIn('images', calls[1]['messages'][-1])
        self.assertEqual(calls[1]['format']['type'], 'object')
        self.assertEqual(calls[1]['format']['properties']['evidence_ids']['items']['enum'], ['E1'])
        self.assertIn(packet['visual_observations'], calls[1]['messages'][-1]['content'])
        self.assertEqual(report['evidence_ids'], ['E1'])

    def test_contact_sheet_preserves_chronology_and_blank_padding(self):
        import base64
        import cv2
        import numpy as np
        from sentinel.context import chronological_sheet
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        images = []
        for color in colors:
            frame = np.full((240, 320, 3), color, dtype=np.uint8)
            images.append(base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode())
        encoded = chronological_sheet(dict(images=images,
            visual_frames=[dict(evidence_id=f'E{i}') for i in range(3)]))[0]
        sheet = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(sheet.shape, (536, 640, 3))
        for point, color in zip([(140,160), (140,480), (408,160)], colors):
            self.assertTrue(np.allclose(sheet[point], color, atol=5))
        self.assertEqual(int(sheet[408,480].sum()), 0)
        self.assertEqual(chronological_sheet(dict(images=[])), [])

    def test_framewise_observations_keep_each_image_separate(self):
        import io,json
        from unittest.mock import patch
        from sentinel.context import assess
        calls=[]
        packet=dict(framewise_review=True,images=['first','second'],
                    evidence=[dict(id='E1',time=1),dict(id='E2',time=2)],
                    visual_frames=[dict(evidence_id='E1'),dict(evidence_id='E2')])
        def respond(request,**kwargs):
            body=json.loads(request.data);calls.append(body)
            content=f'E{len(calls)}: Visible person.' if len(calls)<=2 else json.dumps(reply())
            return io.BytesIO(json.dumps(dict(message=dict(content=content))).encode())
        with patch('sentinel.context.urllib.request.urlopen',respond):
            assess(packet)
        self.assertEqual([c['messages'][0]['images'] for c in calls[:2]],[['first'],['second']])
        self.assertNotIn('images',calls[-1]['messages'][-1])
        self.assertIn('E2: Visible person.',packet['visual_observations'])

    def test_truncated_observation_cannot_become_a_report(self):
        import io,json
        from unittest.mock import patch
        from sentinel.context import assess
        packet=dict(images=['image'],evidence=[dict(id='E1',time=1)])
        truncated=json.dumps(dict(done_reason='length',message=dict(content='Partial'))).encode()
        with patch('sentinel.context.urllib.request.urlopen',side_effect=lambda *a, **k: io.BytesIO(truncated)) as request:
            with self.assertRaisesRegex(ValueError,'truncated'):
                assess(packet)
        self.assertEqual(request.call_count,2)

    def test_brief_action_frames_survive_sampling(self):
        from sentinel.context import select_visual_frames
        frames = [(i, f'E{i}', 'image', 20 if i == 13 else 0) for i in range(30)]
        chosen = select_visual_frames(frames)
        self.assertEqual([f[0] for f in chosen], [0, 7, 13, 22, 29])
        self.assertEqual(len(select_visual_frames(frames[:3])), 3)
        late_burst = [(i, f'E{i}', 'image', 20 if i == 18 else 0) for i in range(21)]
        selected_times = [f[0] for f in select_visual_frames(late_burst)]
        self.assertEqual(selected_times, [0, 5, 10, 18, 20])
        self.assertLessEqual(max(b-a for a, b in zip(selected_times, selected_times[1:])), 8)
        frames = [(i, f'E{i}', 'image', 0) for i in range(30)]
        self.assertEqual([f[0] for f in select_visual_frames(frames)], [0, 7, 14, 22, 29])

    def test_selected_motion_frames_have_matching_evidence(self):
        import numpy as np
        packets = []
        engine = ContextEngine(lambda packet: packets.append(packet) or reply())
        settings = dict(enabled=True, scene='Unknown', recording_start='', expected='Unknown')
        try:
            for t in range(30):
                frame = np.full((20, 20, 3), 255 if t >= 13 else 0, dtype=np.uint8)
                engine.observe(0, 'vid', t, {}, 100, settings, frame=frame)
            engine.observe(0, 'vid', 29, {}, 100, settings, frame=frame, force=True)
            while True:
                result = engine.results.get(timeout=3)
                if result['timestamp'] == 29:
                    final = result['packet']
                    break
            ids = {item['id'] for item in final['evidence']}
            self.assertTrue(all(f['evidence_id'] in ids for f in final['visual_frames']))
            self.assertIn(13, [f['video_seconds'] for f in final['visual_frames']])
            self.assertLessEqual(len(final['images']), 5)
            engine.reset()
            self.assertIsNone(engine.last_gray)
        finally:
            engine.close()

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
