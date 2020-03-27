import numpy as np

"""Models for generating word vectors."""

def sigma(x):
    """sigmoid function"""
    return 1.0 / (np.exp(x) + 1.0)

class SkipGramWV:
    """Associates a vocabulary of V words with a dense (V, d) tensor. d is the dimensionality of
    the word vectors.
        As in the original skipgram papers, two separate "input" and "output" vectors are used for each
        word - this makes gradient computations a little simpler.
        Single word vectors can be obtained after training by concatenating the IO vectors."""

    def __init__(self, vocabulary_size, dimension):
        self.vocabulary_size = vocabulary_size
        if dimension % 2 !=0:
            raise ValueError("Expecting an even vector dimension.")
        self.dimension = dimension
        self.sub_dimension = self.dimension //2
    
        self._weights = None
        self._init_weights()

    def _init_weights(self):
        """Initialize the word vectors."""
        # normalization is to keep the dot products order-1
        self._weights = np.random.randn(self.vocabulary_size, self.dimension) / np.sqrt(self.dimension)

    @property
    def _input_vectors(self):
        """Slice into the weights tensor that defines the input vectors."""
        return self._weights[:, :self.sub_dimension]
    
    @property
    def _output_vectors(self):
        """Slice into the weights tensor that defines the output vectors."""
        return self._weights[:, self.sub_dimension:]

    def _stack_vectors(self, input_index, context_indices, noise_indices):
        """Stacks IO word vectors corresponding to a particular input and context.
            input_index: int, index of the input word in the weights
            context_indices: list of ints, indices of the context words in the weights.
            noise_inidices: list of ints, indices of noise words in the weights.
            
            Returns: (sub_d,) dimensional input vector as numpy array
                    (num_context + num_noise, sub_d) numpy array of output vectors
                        the first num_context rows are the context output vectors
                        the next num_noise worse are the noise output vectors
        """
        input_vec = self._input_vectors[input_index].copy()
        output_vecs = self._output_vectors[context_indices + noise_indices, :]
        return input_vec, output_vecs

    def _gradient_tensors(self, input_index, context_indices, noise_indices):
        """Computes nonzero gradients for the given input, context, and noise samples. 
            Returns: (subdim,) input vector gradient tensor
                      (num_context, subdim) context output vector gradient tensor
                      (num_noise, subdim) noise output vector gradient tensor."""
        
        iv, ov = self._stack_vectors(input_index, context_indices, noise_indices)
        num_context = len(context_indices)
        num_noise = len(noise_indices)
        num_output = num_context + num_noise

        #all required dot products
        io_products = np.dot(ov, iv)
        context_io_products = io_products[:num_context]
        noise_io_products = io_products[num_context:]

        #gradients for the context output vectors
        # (num_context, subdim) tensor
        context_output_grads = np.outer(sigma(-context_io_products), iv) 

        #gradents for the noise output vectors
        # (num_noise, subdim) tensor
        noise_output_grads = np.outer(sigma(noise_io_products), iv)

        # gradient for the input vector
        # (subdim,) tensor
        input_grad_c = - sum([sigma(-context_io_products[i]) * ov[i] for i in range(num_context)])
        input_grad_n = sum([sigma(noise_io_products[i]) * ov[num_context+i] for i in range(num_noise)])
        input_grad = (input_grad_c + input_grad_n).reshape(self.sub_dimension)

        return input_grad, context_output_grads, noise_output_grads

    