import copy,importlib.util,json,unittest,yaml
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('validate_contracts',R/'tools/validate_contracts.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
S=json.loads((R/'SOURCES.json').read_text())
C=yaml.safe_load((R/'contracts/CE.yaml').read_text())
class ContractSchemaTests(unittest.TestCase):
 def test_original(self):self.assertFalse(any(v.validate_all(R)['errors'].values()))
 def test_missing_output_rejected(self):
  c=copy.deepcopy(C);del c['examples'][0]['expected']['fields']['gradient'];self.assertTrue(v.validate_contract(c,S))
 def test_unresolved_basis_rejected(self):
  c=copy.deepcopy(C);c['clauses'][0]['basis']=['missing.quote'];self.assertTrue(v.validate_contract(c,S))
 def test_unknown_profile_rejected(self):
  c=copy.deepcopy(C);c['examples'][0]['profile']='missing';self.assertTrue(v.validate_contract(c,S))
 def test_duplicate_example_rejected(self):
  c=copy.deepcopy(C);c['examples'].append(c['examples'][0]);self.assertTrue(v.validate_contract(c,S))
 def test_executable_expected_rejected(self):
  c=copy.deepcopy(C);c['examples'][0]['expected']['fields']['loss']='__import__("os").getcwd()';self.assertTrue(v.validate_contract(c,S))
 def test_boolean_not_number(self):
  with self.assertRaises(ValueError):v.check_expression(True)
 def test_unreviewed_rejected(self):
  c=copy.deepcopy(C);c['review']['status']='draft';self.assertTrue(v.validate_contract(c,S))
 def test_no_accidental_runtime_certification(self):self.assertFalse(v.validate_all(R)['runtime_ready'])
if __name__=='__main__':unittest.main()
