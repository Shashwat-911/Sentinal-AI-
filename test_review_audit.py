import unittest
from test_context import reply
from sentinel.context_report import review_output
class AuditRegressions(unittest.TestCase):
 def test_missing_earlier_scene_cannot_clear_whole_window(self):
  from sentinel.voting import classify_window
  r=classify_window({'subject_scope':'people','classification':'not_suspicious'}, {'requested_window':{'start':0,'end':29},'evidence':[{'time':28,'people':[{'signals':[]}]}]})
  self.assertEqual(r['classification'],'uncertain')
  self.assertTrue(r['incomplete_window_coverage'])
 def test_denied_force_cannot_raise_suspicion(self):
  r=reply();r['classification']='not_suspicious';r['observed_events']=[{'event':'forceful_object_contact','description':'No visible force or impact; actions are controlled and non-aggressive.','evidence_ids':['E0']}]
  result=review_output(r,{})
  self.assertEqual(result['observed_events'],[]);self.assertEqual(result['classification'],'not_suspicious')
 def test_fire_in_empty_scene_is_safety_event(self):
  r=reply();r.update(subject_scope='empty',classification='not_suspicious',observed_events=[{'event':'fire_or_smoke','description':'Flames spread across the floor.','evidence_ids':['E0']}])
  self.assertEqual(review_output(r,{})['classification'],'safety_concern')
if __name__=='__main__':unittest.main()
