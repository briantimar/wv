from collections import Counter

"""Statistics on the tokenized dataset."""

class TokenSet:
    """Iterates over tokens stored as lines in a text file."""

    def __init__(self, fname):
        self.fname = fname
        self._count = None
        self._multiset = None
        self._sorted_words = None

    def __iter__(self):
        with open(self.fname) as f:
            for i, ln in enumerate(f.readlines()):
                yield ln.strip()
        self._count = i+1

    def _build_multiset(self):
        self._multiset = Counter(iter(self))

    def _sort_words(self):
        counts = self.multiset
        self._sorted_words = sorted(counts.items(), key=lambda t: -t[1])

    @property
    def multiset(self):
        if self._multiset is None:
            self._build_multiset()
        return self._multiset
    
    @property
    def sorted_words(self):
        if self._sorted_words is None:
            self._sort_words()
        return self._sorted_words

    def __len__(self):
        if self._count is None:
            raise ValueError
        return self._count        

if __name__ == "__main__":
    tokenpath = "data/gibbon_daf_tokens.txt"
    tokens = TokenSet(tokenpath)
    sw = tokens.sorted_words