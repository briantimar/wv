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

    def test_apply_vector_updates(self):
        model = SkipGramWV(10, 2)
        model._weights = np.zeros_like(model._weights)

        input_index = 4
        context_indices = [2, 5]
        noise_indices = [3, 3, 0]
        di = np.ones(1)
        dc = np.ones((2,1))
        dn = np.ones((3, 1))

        model._apply_vector_updates(input_index, context_indices, noise_indices,
                                    di, dc, dn)
        
        self.assertAlmostEqual(diff(model._input_vectors[4], -np.ones(1)), 0)
        self.assertAlmostEqual(diff(model._output_vectors[2], -np.ones(1)), 0)
        self.assertAlmostEqual(diff(model._output_vectors[3], - 2 * np.ones(1)), 0)

    def test_do_sgd_update(self):
        input_index = 4
        context_indices = [1,2]
        noise_indices = [4, 12]

        self.model.do_sgd_update(input_index, context_indices, noise_indices, 0.01)

if __name__ == "__main__":
    unittest.main()