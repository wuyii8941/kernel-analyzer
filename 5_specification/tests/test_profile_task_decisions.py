import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry,schema

class ProfileTaskDecisions(unittest.TestCase):
    def setUp(self):
        self.reviewed_files=patch.object(registry,'_validated_file_reasons',return_value=[])
        self.reviewed_files.start();self.addCleanup(self.reviewed_files.stop)

    def contract(self):
        c=registry.load('RANKMEAN')
        c.data=copy.deepcopy(c.data)
        c.data['profiles']['base']['basis']=['RM.TASK']
        return c

    def errors(self,c):
        return schema.validate(c.data,registry.load_sources())

    def test_dependent_profile_cannot_turn_off_task_choice(self):
        c=self.contract();c.data['profiles']['base']['task_choice_required']=False
        self.assertTrue(any('RM.TASK' in e and 'task_choice_required' in e for e in self.errors(c)))
        ok,reasons,_=registry.activation(c,'base')
        self.assertFalse(ok)
        self.assertTrue(any('RM.TASK' in e and 'task_choice_required' in e for e in reasons))

    def test_task_decision_must_identify_its_profiles(self):
        c=self.contract();c.data['profiles']['base']['basis']=[]
        self.assertTrue(any('RM.TASK' in e and 'profile' in e for e in self.errors(c)))

    def test_indirect_dependency_requires_task_choice(self):
        c=self.contract();c.data['profiles']['base']['basis']=['RM.ZERO']
        c.data['profiles']['base']['task_choice_required']=False
        self.assertTrue(any('RM.TASK' in e and 'task_choice_required' in e for e in self.errors(c)))
        c.data['profiles']['base']['task_choice_required']=True
        self.assertEqual(self.errors(c),[])

    def test_unrelated_ce_profile_needs_no_task_declaration(self):
        c=registry.load('CE')
        c.data['profiles']['index_smoothing_R_A']['basis']=['CE.R_A']
        ok,reasons,_=registry.activation(c,'index_unsmoothed')
        self.assertTrue(ok,reasons)
        ok,reasons,_=registry.activation(c,'index_smoothing_R_A')
        self.assertFalse(ok)
        self.assertTrue(any('explicit task binding' in e for e in reasons))

if __name__=='__main__':unittest.main()
