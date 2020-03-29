import unittest
from tokens import  TokenSet, ContextIterator

class TestTokenSet(unittest.TestCase):
    
    def setUp(self):
        tokenpath = "data/gibbon_daf_tokens.txt"
        self.num_word = 1000
        self.doc = TokenSet(tokenpath, num_word=self.num_word)
    
    def test_len(self):
        self.assertEqual(self.doc.vocab_size, self.num_word)
        self.assertAlmostEqual(sum(self.doc.probs), 1.0)
        self.assertEqual(len(self.doc.probs), self.doc.num_word)


class TestContextIterator(unittest.TestCase):

    def test_iter(self):

        class DummySet:

            def __init__(self, words):
                self.words = words
            def indices(self):
                for word in self.words:
                    yield word
            def sample_noise(self, N, distribution=None):
                return [5]*N

        ts = DummySet(list(range(4)))
        ci = ContextIterator(ts, context_radius=1)
        inputs = []
        contexts = []
        for input, context, __ in ci:
            inputs.append(input)
            contexts.append(context)
        self.assertEqual(inputs, list(range(4)))
        self.assertEqual(contexts, [[1], [0,2], [1, 3], [2]])



if __name__ == "__main__":
    unittest.main()