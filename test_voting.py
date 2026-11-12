import unittest
from sentinel.voting import classify_window


class VotingTests(unittest.TestCase):
    def test_pedestrians_cannot_clear_vehicle_collision(self):
        result = classify_window({'subject_scope': 'vehicles', 'classification': 'not_suspicious'},
            {'evidence': [{'people': [{'signals': []}]}]})
        self.assertIsNone(result['rule_vote'])
        self.assertEqual(result['classification'], 'uncertain')

    def test_ai_weightage_drives_suspicious_over_empty_rules(self):
        result = classify_window({'subject_scope': 'people', 'classification': 'suspicious'},
            {'evidence': [{'people': [{'signals': []}]}]})
        self.assertEqual(result['ai_weight'], 0.8)
        self.assertEqual(result['rule_weight'], 0.2)
        self.assertEqual(result['concern_vote_percent'], 80)
        self.assertEqual(result['classification'], 'suspicious')

    def test_safety_survives_abstention(self):
        result = classify_window({'subject_scope': 'vehicles', 'classification': 'uncertain'},
            {'evidence': [], 'safety_event': True})
        self.assertEqual(result['classification'], 'safety_concern')

    def test_ai_not_suspicious_overrides_minor_rule_warning(self):
        result = classify_window({'subject_scope': 'people', 'classification': 'not_suspicious'},
            {'evidence': [{'people': [{'signals': ['REPEATED_MOVEMENT'], 'behavior': 'MEDIUM'}]}],
             'rule_assessment': 'review'})
        self.assertEqual(result['rule_vote'], 1)
        self.assertEqual(result['ai_vote'], 0)
        self.assertEqual(result['concern_vote_percent'], 20)
        self.assertEqual(result['classification'], 'not_suspicious')

    def test_vote_summary_reflects_ai_weighted_alignment(self):
        from sentinel.voting import vote_summary
        summary = vote_summary({'subject_scope': 'people', 'classification': 'not_suspicious'},
            {'evidence': [{'people': [{'signals': ['REPEATED_MOVEMENT']}]}],
             'rule_assessment': 'review'})
        self.assertIn('COMBINED: NOT SUSPICIOUS (20%)', summary)
        self.assertIn('AI 80%: no concern', summary)
        self.assertIn('Rules 20%: concern', summary)

    def test_risk_and_overlay_align_with_ai_context(self):
        import numpy as np
        from sentinel.risk import calculate_entity_risk
        from sentinel.alerting import draw_overlays
        track = {
            'track_id': 2,
            'class_name': 'person',
            'bbox': [10, 10, 80, 180],
            'active_threats': {'REPEATED_MOVEMENT'},
            'behavior_state': 'MEDIUM',
            'suspicion_score': 60,
        }
        # Without AI context: MEDIUM (60%)
        score_raw, sev_raw, _ = calculate_entity_risk(track)
        self.assertEqual(score_raw, 60)
        self.assertEqual(sev_raw, 'MEDIUM')

        # When AI says not_suspicious: aligned to NORMAL (12%) and alerts suppressed
        score_ai_ok, sev_ai_ok, _ = calculate_entity_risk(track, ai_context={'classification': 'not_suspicious'})
        self.assertEqual(score_ai_ok, 12)
        self.assertEqual(sev_ai_ok, 'NORMAL')
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        _, alerts = draw_overlays(frame, {'p2': track}, ai_context={'classification': 'not_suspicious'}, run_behavior=False)
        self.assertEqual(alerts, [])

        # When AI says suspicious: elevated to HIGH (80%)
        score_ai_bad, sev_ai_bad, _ = calculate_entity_risk(track, ai_context={'classification': 'suspicious'})
        self.assertEqual(score_ai_bad, 80)
        self.assertEqual(sev_ai_bad, 'HIGH')

        # Safety fall is never suppressed by AI not_suspicious
        fall_track = dict(track, active_threats={'POSSIBLE_FALL'})
        _, sev_fall, _ = calculate_entity_risk(fall_track, ai_context={'classification': 'not_suspicious'})
        self.assertEqual(sev_fall, 'MEDIUM')

    def test_door_access_attempt_upgrades_to_suspicious(self):
        from test_context import reply
        from sentinel.context_report import review_output
        r = reply()
        r['classification'] = 'not_suspicious'
        r['behavior_summary'] = 'A single person approaches a door, attempts to open it, and then steps back.'
        res = review_output(r, {'visual_observations': 'person attempts to open the door'})
        self.assertEqual(res['classification'], 'suspicious')
