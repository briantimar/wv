from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

"""Statistics on the tokenized dataset."""

class TokenSet:
    """Iterates over tokens stored as lines in a text file."""

    def __init__(self, fname):
        self.fname = fname
        self._count = None
        self._multiset = None
        self._sorted_words = None
        self._index_map = None

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

    def _build_index_map(self):
        self._index_map = {word_tup[0] : i for i, word_tup in enumerate(self.sorted_words)}

    @property
    def multiset(self):
        """a dictionary mapping words to frequencies."""
        if self._multiset is None:
            self._build_multiset()
        return self._multiset
    
    @property
    def sorted_words(self):
        """A list of (word, count) pairs in descending order of count."""
        if self._sorted_words is None:
            self._sort_words()
        return self._sorted_words
    
    @property
    def index_map(self):
        """Dictionary mapping word strings to unique indices. These correspond to order when 
        sorted by descending frequency."""
        if self._index_map is None:
            self._build_index_map()
        return self._index_map

    def __len__(self):
        if self._count is None:
            raise ValueError
        return self._count  

    def indices(self):
        """Iterator over indices of words in the training set. These are defined by frequency, namely as the
        index of a word in the full list of words sorted by frequency."""
        for word in iter(self):
            yield self.index_map[word]

def write_token_stats(tokenpath, statsfile):
        tokens = TokenSet(tokenpath)
        sw = tokens.sorted_words
        with open(statsfile, 'w') as f:
            f.write(f"Token statistics from {tokenpath}\n")
            f.write(f"Total token count: {len(tokens)}\n")
            f.write("tokens sorted by count follow\n")
            f.write("Word, count, probability\n")
            f.write('-'*20 + "\n")

            for word, count in sw:
                f.write(f"{word}, {count}, {count/len(tokens):.6e}\n")

def plot_token_probs(tokenpath):
    tokens = TokenSet(tokenpath)
    sw = tokens.sorted_words

    min_ct = 5
    counts_all = np.asarray([t[1] for t in sw])

    counts = counts_all[counts_all>=min_ct]
    probabilities = counts / len(tokens)
    probabilities_all = counts_all / len(tokens)

    H = -np.sum(probabilities_all * np.log2(probabilities_all))

    fig, ax = plt.subplots()
    ax.plot(np.log(probabilities)/np.log(10), '-o')
    ax.set_xlabel("token index")
    ax.set_ylabel("log10 p(word)")
    ax.set_title(f"token probs, min count = {min_ct}. H={H:.3f} bits")
    plt.savefig("token_probs.png")

if __name__ == "__main__":
    tokenpath = "data/gibbon_daf_tokens.txt"
    statsfile = "data/token_stats.txt"

    # plot_token_probs(tokenpath)


   


