import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/mx_boundary'
spec=importlib.util.spec_from_file_location('mx_evidence_verify',ROOT/'verify.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class MXDeviceEvidence(unittest.TestCase):
    def setUp(self):self.record=json.loads((ROOT/'observed.json').read_text())

    def test_independent_calculation_preserves_all_failed_coordinates(self):
        result=audit.verify(self.record)
        self.assertEqual(result['cases_with_element_difference'],3)
        self.assertTrue(all(r['same_on_both_replays'] and r['scale_matches'] for r in result['cases']))
        for row in result['cases'][3:]:
            self.assertEqual(row['different_element_coordinates'],list(range(32)))
            self.assertEqual(row['different_dequantized_coordinates'],list(range(32)))

    def test_expected_values_are_not_trusted(self):
        self.record['cases'][-1]['expected']['elements']=['0']*32
        with self.assertRaises(ValueError):audit.verify(self.record)

    def test_missing_bytes_cases_or_replays_reject(self):
        for kind in ('case','byte','replay'):
            record=copy.deepcopy(self.record)
            if kind=='case':record['cases'].pop()
            elif kind=='byte':record['cases'][0]['runs'][0]['packed_bytes'].pop()
            else:record['cases'][0]['runs'].pop()
            with self.assertRaises(ValueError):audit.verify(record)

    def test_flags_and_decoded_values_are_not_trusted(self):
        for field,value in (('complete_elements_match',True),('elements',['1']*32),('scale_code',True)):
            record=copy.deepcopy(self.record);record['cases'][-1]['runs'][0][field]=value
            with self.assertRaises(ValueError):audit.verify(record)

if __name__=='__main__':unittest.main()
