import unittest
import numpy as np
from model import SkipGramWV

def diff(array1, array2):
    return np.sum((array1 - array2)**2)

class SkipGramWVTest(unittest.TestCase):

    def setUp(self):
        self.model = SkipGramWV(20, 10)

    def test_stack_vectors(self):
        input_index = 3
        context_indices = [4, 5]
        noise_indices = [2, 2, 7]
        iv, ov = self.model._stack_vectors(input_index, context_indices, noise_indices)

        self.assertEqual(iv.shape, (self.model.sub_dimension,))
        self.assertEqual(ov.shape, (5, self.model.sub_dimension))

        self.assertAlmostEqual(diff(ov[2], ov[3]), 0)

    def test_gradient_tensors(self):
        input_index = 4
        context_indices = [1,2]
        noise_indices = [4, 12]
        ig, cg, ng = self.model._gradient_tensors(input_index, context_indices, noise_indices)
        
        self.assertEqual(ig.shape, (self.model.sub_dimension,))
        self.assertEqual(cg.shape, (len(context_indices), self.model.sub_dimension))
        self.assertEqual(ng.shape, (len(noise_indices), self.model.sub_dimension))

if __name__ == "__main__":
    unittest.main()