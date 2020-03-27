from collections import Counter, deque
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

"""Statistics on the tokenized dataset."""

class TokenSet:
    """Iterates over tokens stored as lines in a text file."""

    def __init__(self, fname, noise_distribution="frequency",noise_refresh_size=None):
        """Noise distribution: distribution from which to draw noise tokens"""

        self.fname = fname
        self._count = None
        self._multiset = None
        self._sorted_words = None
        self._index_map = None

        self.noise_distribution = noise_distribution
        self.noise_refresh_size = noise_refresh_size
        self._noise_samples = []
        self._init()

    def _init(self):
        """Performs all "startup" functions. Builds multiset and frequency-sorted words, and pre-computes
        a batch of noise words."""
        self._build_multiset()
        self._sort_words()
        self._build_index_map()
        if self.noise_refresh_size is None:
            print("No refresh size provided, using default 50000")
            self.noise_refresh_size = 50000
        self._noise_samples = self._sample_noise(self.noise_refresh_size)

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

    def _sample_frequency(self, N):
        """Sample N word indices drawn according to the frequency distribution defined by the dataset."""
        return list(np.random.choice(len(self.sorted_words), size=(N,),p=self.probs))

    def _sample_noise(self, N):
        """Computes N "noise" word indices."""
        if self.noise_distribution == "frequency":
            return self._sample_frequency(N)
        else:
            raise ValueError(f"Invalid noise distribution: {self.noise_distribution}")

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

    @property
    def counts(self):
        """array of counts in descending order."""
        return np.asarray([word_tup[1] for word_tup in self.sorted_words])

    @property
    def num_tokens(self):
        """Number of unique tokens in the dataset."""
        return len(self.sorted_words)

    @property
    def probs(self):
        """Returns array of probabilities, in descending order, for words in the dataset."""
        return self.counts / len(self)

    def indices(self):
        """Iterator over indices of words in the training set. These are defined by frequency, namely as the
        index of a word in the full list of words sorted by frequency."""
        for word in iter(self):
            yield self.index_map[word]

    def sample_noise(self, N):
        """Return list of N word indices drawn from the 'noise' distribution."""
        while len(self._noise_samples) < N:
            self._noise_samples += self._sample_noise(self.noise_refresh_size)
        noise_words, self._noise_samples = self._noise_samples[:N], self._noise_samples[N:]
        return noise_words

    def __len__(self):
        if self._count is None:
            raise ValueError
        return self._count  

class ContextIterator:
    """ A wrapper around tokenset that yields input indices along with the indices of tokens within a fixed context
    window."""

    def __init__(self, tokenset, context_radius, num_noise=None):
        """context_radius: how far to search forwards and backwards from the input word to define context."""
        if context_radius <1:
            raise ValueError(f"Not a valid context radius: {context_radius}")
        self.context_radius = context_radius
        # total size of the context region, including the input
        self.context_size = 2 * context_radius + 1
        self.tokenset = tokenset
        
        if num_noise is None:
            num_noise = 2 * self.context_radius
        self.num_noise = num_noise

        # all words, including the input, in the context region
        self._context = deque(maxlen=self.context_size)

    def get_noise_indices(self, num_noise):
        """Returns num_noise word indices sampled from the underlying tokenset distribution."""
        return list(self.tokenset.sample_noise(num_noise))

    def __iter__(self):
        """Iterate over (input_index, context_indices, noise_indices) tuples)
            Assumes the dataset has size at least context_size"""
        self._context.clear()
        # buffer one edge of the word-stream with None, to define contexts missing part of the right side
        index_iter = chain(self.tokenset.indices(), [None]*self.context_radius)
        for i, index in enumerate(index_iter):
            self._context.append(index)
            if i >= self.context_radius:
                #available context has been added
                input_loc = len(self._context) -1 - self.context_radius
                _context_all = list(self._context)
                if _context_all[-1] is None:
                    end = _context_all.index(None)
                    _context_all = _context_all[:end]
                context_indices = _context_all[:input_loc] + _context_all[input_loc+1:]
                noise_indices = self.get_noise_indices(self.num_noise)
                yield (_context_all[input_loc], context_indices, noise_indices)


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

    tokenset = TokenSet(tokenpath)
    print(tokenset.sample_frequency(100))

   


