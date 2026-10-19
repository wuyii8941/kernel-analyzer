import sys,unittest,tempfile,types
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import v24,registry
from kernel_spec.num import NotEstablished

class V24ModuleOrigin(unittest.TestCase):
    def test_cached_same_name_module_is_not_a_reference(self):
        foreign=types.ModuleType('spec_cross_entropy');foreign.__file__='/unrelated/spec_cross_entropy.py'
        with patch.dict(sys.modules,{'spec_cross_entropy':foreign}):
            with self.assertRaises(NotEstablished):v24.module('spec_cross_entropy')

    def test_shadow_module_is_rejected_before_its_code_runs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'.cache') as temp:
            path=Path(temp)/'spec_ema.py';path.write_text('raise AssertionError("shadow module executed")\n')
            with patch.dict(sys.modules),patch.object(sys,'path',[temp]+sys.path):
                sys.modules.pop('spec_ema',None)
                with self.assertRaises(NotEstablished):v24.module('spec_ema')

    def test_dependency_conflict_is_rejected_even_with_cached_parent(self):
        v24.module('spec_optimizers')
        foreign=types.ModuleType('rigorous');foreign.__file__='/unrelated/rigorous.py'
        with patch.dict(sys.modules,{'rigorous':foreign}):
            with self.assertRaises(NotEstablished):v24.module('spec_optimizers')

    def test_only_declared_modules_are_available_and_valid_origin_still_works(self):
        with self.assertRaises(NotEstablished):v24.module('json')
        expected={'spec_cross_entropy':'phase1','spec_optimizers':'phase2','spec_training_program':'phase2',
                  'spec_ema':'phase3','rigorous':'phase2','rigorous_mp':'phase2'}
        for name,phase in expected.items():
            self.assertEqual(Path(v24.module(name).__file__).resolve(),(v24.ALG/phase/f'{name}.py').resolve())

    def test_runtime_query_reports_unavailable_reference_on_import_conflict(self):
        foreign=types.ModuleType('spec_cross_entropy');foreign.__file__='/unrelated/spec_cross_entropy.py'
        with patch.dict(sys.modules,{'spec_cross_entropy':foreign}):
            result=registry.query('CE','index_unsmoothed',{})
            self.assertEqual(result.outcome,'not_established')
            self.assertIn('spec_cross_entropy',result.note)
            self.assertEqual(result.labels['semantic_verdict'],'not_established')

if __name__=='__main__':unittest.main()
