import unittest
from tokens import TokenSet, ContextIterator

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