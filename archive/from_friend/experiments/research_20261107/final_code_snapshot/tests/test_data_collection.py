import unittest
from unittest.mock import patch
from data_collection.validate import concentration,validate
from data_collection.chembl import paginate

class DataChecks(unittest.TestCase):
 def setUp(self):
  self.row=dict(SMILES='CCO',source_url='https://example.org/study',ic50_value='1000',ic50_units='nM',ic50_relation='=',cc50_value='0.01',cc50_units='mM',cc50_relation='=',culture='Vero',strain='VACV',compatibility_evidence='Same experiment, verified from paper',reviewer='reviewer',review_status='approved')
 def test_units_and_ratio(self):
  row=validate(self.row); self.assertEqual(row['IC50_uM_VACV'],1); self.assertEqual(row['CC50_uM'],10); self.assertEqual(row['SI VACV'],10)
 def test_reject_censored(self):
  self.row['ic50_relation']='>'; self.assertRaises(ValueError,validate,self.row)
 def test_reject_unreviewed(self):
  self.row['review_status']='pending'; self.assertRaises(ValueError,validate,self.row)
 def test_reject_test_structure_even_if_approved(self):
  self.assertRaisesRegex(ValueError,'test set',validate,self.row,{'OCC'})
 def test_reject_unsupported_units(self): self.assertRaises(ValueError,concentration,1,'mg/L','=')
 def test_reject_bad_smiles(self):
  self.row['SMILES']='not smiles'; self.assertRaises(ValueError,validate,self.row)
 def test_pagination(self):
  pages=[{'page_meta':{'next':'/chembl/api/data/assay.json?offset=100'},'assays':[1]},{'page_meta':{'next':None},'assays':[2]}]
  with patch('data_collection.chembl.get_json',side_effect=pages):
   results=list(paginate('assay',{},'assays',3)); self.assertEqual([r[1]['assays'][0] for r in results],[1,2])
 def test_refuses_incomplete_pagination(self):
  with patch('data_collection.chembl.get_json',return_value={'page_meta':{'next':'?offset=100'}}):
   with self.assertRaises(RuntimeError): list(paginate('assay',{},'assays',1))
if __name__=='__main__': unittest.main()
