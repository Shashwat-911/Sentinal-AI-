import tempfile
import unittest
from pathlib import Path
from sentinel.incident_reports import persist, load_reports, matches, HypothesisHistory
from sentinel.context_report import validate_report, format_brief
from test_context import reply

class ReportTests(unittest.TestCase):
    def test_verbose_prose_does_not_discard_a_valid_report(self):
        from sentinel.context_report import limit_report_prose
        report = reply()
        report['action_sequence'] = [str(i) for i in range(8)]
        report['scene_analysis'] = ['detail'] * 5
        report = limit_report_prose(report)
        validate_report(report, dict(evidence=[dict(id='E1')]))
        self.assertEqual(report['action_sequence'], ['0', '1', '2', '7'])
        self.assertEqual(len(report['scene_analysis']), 2)
        report['scene_analysis'] = [42] * 5
        with self.assertRaises(ValueError):
            validate_report(limit_report_prose(report), dict(evidence=[dict(id='E1')]))

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
        report["missing_information"] = ["Identity of the person", "Occupation", "Whether the door opens"]
        report['concern_factors'] = ['No visible weapons or tools.', 'Person pulls another person to the ground.']
        report["reassuring_factors"] = ["No other individuals present", "Person leaves the object undamaged",
                                       "No visible weapons or tools are present."]
        reviewed = review_output(report)
        self.assertEqual(reviewed["classification"], "uncertain")
        self.assertEqual(reviewed["lighting_assessment"]["appearance"], "indeterminate")
        self.assertEqual(reviewed["missing_information"], ["Whether the door opens"])
        self.assertEqual(len(reviewed["reassuring_factors"]), 1)
        self.assertEqual(reviewed['concern_factors'], ['Person pulls another person to the ground.'])

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

    def test_animal_scope_cannot_populate_human_task_peaks(self):
        from sentinel.context_report import review_output
        report = reply()
        report['subject_scope'] = 'animals_only'
        report['classification'] = 'not_suspicious'
        report['concern_factors'] = ['None', 'None']
        report = review_output(report)
        self.assertEqual(report['plausible_tasks'], [])
        self.assertEqual(report['concern_factors'], [])
        self.assertEqual(report['missing_information'], [])
        history = HypothesisHistory()
        history.update(report, 10)
        self.assertEqual(history.items, {})

    def test_duplicate_family_keeps_strongest_without_adding_estimates(self):
        from sentinel.context_report import review_output
        report = reply()
        report['plausible_tasks'][1]['family'] = 'waiting'
        report['plausible_tasks'][1]['probability_percent'] = 60
        validate_report(report, dict(evidence=[dict(id='E1')]))
        report = review_output(report)
        self.assertEqual(len(report['plausible_tasks']), 1)
        self.assertEqual(report['plausible_tasks'][0]['probability_percent'], 60)

    def test_shaded_daylight_and_artificial_light_do_not_set_clock(self):
        from sentinel.context_report import review_output, lighting_text
        packet = dict(evidence=[dict(brightness_0_255=40)], capture_time='unknown')
        for appearance, reason in [('daylight_like', 'Sunlit lawn beyond shaded foreground'),
                                   ('artificial_light', 'Visible ceiling lamps illuminate the room')]:
            report = reply()
            report['lighting_assessment'] = dict(appearance=appearance, reason=reason)
            report = review_output(report)
            self.assertEqual(report['lighting_assessment']['appearance'], appearance)
            self.assertNotIn('night', lighting_text(report, packet))
            self.assertEqual(packet['capture_time'], 'unknown')

    def test_lighting_uses_fixture_and_visibility_cues_not_brightness_inference(self):
        from sentinel.context_report import review_output
        report = reply()
        report['lighting_assessment'] = dict(appearance='indeterminate', reason='Only image intensity was cited')
        reviewed = review_output(report, {'visual_observations': 'Lighting is from overhead fluorescent fixtures.'})
        self.assertEqual(reviewed['lighting_assessment']['appearance'], 'artificial_light')
        report = reply()
        report['lighting_assessment'] = dict(appearance='indeterminate', reason='Blue-toned scene')
        reviewed = review_output(report, {'visual_observations': 'Road is dark; lighting is dim and blue-toned.'})
        self.assertEqual(reviewed['lighting_assessment']['appearance'], 'low_visibility')
        report = reply()
        report['lighting_assessment'] = dict(appearance='indeterminate', reason='Distant light is ambiguous')
        reviewed = review_output(report, {'visual_observations': 'Dim blue-toned lighting; distant glowing orb may be a vehicle headlight or streetlamp.'})
        self.assertEqual(reviewed['lighting_assessment']['appearance'], 'low_visibility')
        report = reply()
        report['lighting_assessment'] = dict(appearance='low_visibility', reason='Dim scene')
        packet = {'evidence': [{'brightness_0_255': 113}],
                  'visual_observations': 'Person wears dark pants; lighting suggests daylight.'}
        reviewed = review_output(report, packet)
        self.assertEqual(reviewed['lighting_assessment']['appearance'], 'indeterminate')
        report = reply()
        report['lighting_assessment'] = dict(appearance='daylight_like',
            reason='Sunlit outdoor cues inferred from brightness and shadows.')
        reviewed = review_output(report, {'visual_observations': 'Bright tiled floor; no sun or daylight source visible.'})
        self.assertEqual(reviewed['lighting_assessment']['appearance'], 'indeterminate')

    def test_safety_hypotheses_and_legacy_report_rendering(self):
        report = reply()
        report['classification'] = 'safety_concern'
        report['plausible_tasks'][0]['family'] = 'physical_conflict'
        report['plausible_tasks'][1]['family'] = 'assistance'
        validate_report(report, dict(evidence=[dict(id='E1')]))
        report.pop('subject_scope')
        report['lighting_assessment']['appearance'] = 'artificial_or_night_like'
        self.assertIn('legacy assessment', format_brief(report, {}, 10))

    def test_visible_events_require_review_without_weapons_or_damage(self):
        from sentinel.context_report import review_output
        for event, description, expected in [
                ('forceful_object_contact', 'Person strikes a door with visible recoil.', 'suspicious'),
                ('physical_struggle_or_restraint', 'Two people grapple; one restrains the other.', 'safety_concern'),
                ('person_fall', 'Person loses balance and falls to the ground.', 'safety_concern')]:
            report = reply()
            report['classification'] = 'not_suspicious'
            report['observed_events'] = [dict(event=event, description=description, evidence_ids=['E1'])]
            validate_report(report, dict(evidence=[dict(id='E1')]))
            reviewed = review_output(report)
            self.assertEqual(reviewed['classification'], expected)
            self.assertIn('Initial model label: not suspicious', reviewed['explanation'])
            report['observed_events'][0]['evidence_ids'] = ['invented']
            with self.assertRaises(ValueError):
                validate_report(report, dict(evidence=[dict(id='E1')]))
            report['observed_events'][0]['evidence_ids'] = ['E2']
            with self.assertRaises(ValueError):
                validate_report(report, dict(evidence=[dict(id='E1'), dict(id='E2')],
                                             visual_frames=[dict(evidence_id='E1')]))

    def test_no_events_or_animal_scope_cannot_trigger_review_policy(self):
        from sentinel.context_report import review_output
        report = reply()
        report['classification'] = 'not_suspicious'
        self.assertEqual(review_output(report)['classification'], 'not_suspicious')
        report['subject_scope'] = 'animals_only'
        report['observed_events'] = [dict(event='person_fall', description='Invalid human event', evidence_ids=['E1'])]
        self.assertEqual(review_output(report)['observed_events'], [])

    def test_vehicle_collision_gets_safety_review_without_visible_driver(self):
        from sentinel.context_report import review_output
        report = reply()
        report.update(subject_scope='vehicles', classification='not_suspicious',
                      observed_events=[dict(event='vehicle_collision', description='Two cars collide at the junction', evidence_ids=['E1'])])
        validate_report(report, dict(evidence=[dict(id='E1')]))
        self.assertEqual(review_output(report)['classification'], 'safety_concern')
        report.update(classification='not_suspicious', observed_events=[dict(event='vehicle_collision',
                      description='No visible collision; vehicles pass normally', evidence_ids=['E1'])])
        self.assertEqual(review_output(report)['classification'], 'not_suspicious')

    def test_final_ordinary_window_does_not_clear_prior_review(self):
        from sentinel.incident_reports import session_review_summary
        incident, ordinary = reply(), reply()
        incident.update(classification='safety_concern', behavior_summary='Vehicle impact')
        ordinary.update(classification='not_suspicious', behavior_summary='Empty pavement')
        text = session_review_summary([dict(timestamp=10, assessment=incident), dict(timestamp=30, assessment=ordinary)])
        self.assertIn('Through 10.0s', text)
        self.assertIn('Vehicle impact', text)
        self.assertNotIn('Through 30.0s', text)
        self.assertIn('No review flags', session_review_summary([dict(timestamp=30, assessment=ordinary)]))

    def test_landing_after_dismount_is_not_automatically_a_fall(self):
        from sentinel.context_report import review_output
        report=reply()
        report.update(classification='not_suspicious', observed_events=[dict(event='person_fall',
            description='Person dismounts and lands on the ground.', evidence_ids=['E1'])])
        self.assertEqual(review_output(report)['classification'],'not_suspicious')
        report['observed_events']=[dict(event='person_fall',description='Person jumps, loses balance and falls onto their back.',evidence_ids=['E1'])]
        self.assertEqual(review_output(report)['classification'],'safety_concern')

    def test_explicitly_absent_events_never_raise_a_review_alert(self):
        from sentinel.context_report import review_output
        report = reply()
        report['classification'] = 'not_suspicious'
        report['observed_events'] = [
            dict(event='forceful_object_contact', description='No visible impact or forceful contact with the door.', evidence_ids=['E1']),
            dict(event='forceful_object_contact', description='No visible striking or impact against an object.', evidence_ids=['E1']),
            dict(event='physical_struggle_or_restraint', description='No physical struggle or restraint observed.', evidence_ids=['E1']),
            dict(event='person_fall', description='No fall is visible.', evidence_ids=['E1'])]
        reviewed = review_output(report)
        self.assertEqual(reviewed['classification'], 'not_suspicious')
        self.assertEqual(reviewed['observed_events'], [])
        report['observed_events'] = [dict(event='forceful_object_contact',
            description='Person strikes the door without visible damage.', evidence_ids=['E1'])]
        self.assertEqual(review_output(report)['classification'], 'suspicious')

    def test_non_forceful_item_removal_is_not_mislabeled_as_forceful_event(self):
        from sentinel.context_report import review_output
        report = reply()
        report['classification'] = 'not_suspicious'
        report['observed_events'] = [dict(event='forceful_object_contact',
            description='Person grasps and pulls a handbag from a vehicle seat.', evidence_ids=['E1'])]
        reviewed = review_output(report)
        self.assertEqual(reviewed['classification'], 'not_suspicious')
        self.assertEqual(reviewed['observed_events'], [])
        report['observed_events'] = [dict(event='forceful_object_contact',
            description='Person forcefully strikes the vehicle window and recoils.', evidence_ids=['E1'])]
        self.assertEqual(review_output(report)['classification'], 'suspicious')

    def test_partial_reach_through_vehicle_window_is_not_reported_as_entry(self):
        from sentinel.context_report import review_output
        report = reply()
        report['behavior_summary'] = 'A person enters the vehicle, takes a bag, and exits.'
        report['action_sequence'] = ['Person enters vehicle via window', 'Person exits vehicle']
        packet = {'visual_observations': 'A person is outside the driver side window with one arm extended into the vehicle.'}
        reviewed = review_output(report, packet)
        rendered = ' '.join([reviewed['behavior_summary'], *reviewed['action_sequence']])
        self.assertNotIn('enters the vehicle', rendered.lower())
        self.assertNotIn('exits the vehicle', rendered.lower())
        self.assertIn('reaches into', rendered.lower())
        self.assertIn('leaves the frame', rendered.lower())

    def test_vehicle_item_removal_is_reviewed_without_claiming_theft(self):
        from sentinel.context_report import review_output
        report = reply()
        report['classification'] = 'not_suspicious'
        report['behavior_summary'] = 'A person reaches through a car window and removes a handbag.'
        report['action_sequence'] = ['Person grasps and removes a handbag from a vehicle.']
        report['plausible_tasks'] = [dict(family='property_removal', task='Handbag removed from vehicle',
            probability_percent=70, plausibility='better_supported', support='Handbag is removed from the seat',
            limitations='Ownership and permission are unknown')]
        packet = {'visual_observations': 'A person outside the car reaches in and removes a handbag from the seat.'}
        reviewed = review_output(report, packet)
        self.assertEqual(reviewed['classification'], 'suspicious')
        self.assertIn('ownership', reviewed['explanation'].lower())
        self.assertNotIn('theft', reviewed['explanation'].lower())

if __name__ == "__main__":
    unittest.main()
