import unittest
import numpy as np
from versions.test.training import graph, GraphNet, batch
import torch

class GraphChecks(unittest.TestCase):
 def test_graph_is_symmetric_and_finite(self):
  x,a=graph('CCO'); self.assertEqual(x.shape,(3,10)); np.testing.assert_allclose(a,a.T); self.assertTrue(np.isfinite(a).all()); self.assertTrue((a.diagonal()>0).all())
 def test_invalid_smiles_is_rejected(self):
  with self.assertRaises(ValueError): graph('not valid')
 def test_prediction_independent_of_padding(self):
  torch.manual_seed(3); model=GraphNet();model.eval();graphs=[graph('CCO'),graph('CCCCCC')]
  with torch.no_grad():
   solo=model(*batch(graphs,[0],None))[0]; combined=model(*batch(graphs,[0,1],None))[0]
  torch.testing.assert_close(solo,combined,rtol=1e-6,atol=1e-6)
if __name__=='__main__': unittest.main()
