import unittest
import numpy as np
from model import SkipGramWV, sigma

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


class TestNumerics(unittest.TestCase):


    def test_sigma(self):
        inputs = [-np.log(2), 0, np.log(2)]
        outputs = [1.0/3, .5, 2.0/3 ]
        for i, o in zip(inputs, outputs):
            self.assertAlmostEqual(o, sigma(i))

    def do_grad_check(self, which, index, eps=1e-5):
        """Perform a numerical gradient check with a small model
            which = input, context, or noise
                'input'-> checks grad of input vector. 
                'context' -> checks grad wrt a context output vector
                'noise' -> checks grad wrt a noise output vector
            index = 0, 1. Pick a context or noise vector. If input, just set to zero."""
        vocab_size = 5
        vector_dim = 2
        model = SkipGramWV(vocab_size, vector_dim)

        input_index = 2
        context_indices = [1, 3]
        noise_indices = [0, 4]
        ig, cg, ng = model._gradient_tensors(input_index, context_indices, noise_indices)
        l1 = model.neg_loss(input_index, context_indices, noise_indices)

        if which == "context":
            model._output_vectors[context_indices[index]][0] += eps
            agrad = cg[index,0]
        elif which == "noise":
            model._output_vectors[noise_indices[index]][0] += eps
            agrad = ng[index, 0]
        elif which == "input":
            model._input_vectors[input_index][0] += eps
            agrad = ig[0]
        l2 = model.neg_loss(input_index, context_indices, noise_indices)
        ngrad = (l2 - l1) / eps
        return agrad, ngrad

    def test_context_output_grads(self):
        eps=1e-5
        agrad, ngrad = self.do_grad_check("context", 0, eps=eps)
        self.assertAlmostEqual(agrad, ngrad, delta=10*eps)
        agrad, ngrad = self.do_grad_check("context", 1, eps=eps)
        self.assertAlmostEqual(agrad, ngrad, delta=10*eps)


    def test_noise_output_grads(self):
        eps=1e-5
        agrad, ngrad = self.do_grad_check("noise", 0, eps=eps)
        self.assertAlmostEqual(agrad, ngrad, delta=10*eps)
        agrad, ngrad = self.do_grad_check("noise", 1, eps=eps)
        self.assertAlmostEqual(agrad, ngrad, delta=10*eps)

    def test_input_grads(self):
        eps=1e-5
        agrad, ngrad = self.do_grad_check("input",0, eps=eps)
        self.assertAlmostEqual(agrad, ngrad, delta=10*eps)

if __name__ == "__main__":
    unittest.main()