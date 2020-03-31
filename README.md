# wv

Some python code for computing word vectors from scratch on Gibbon's *Decline and Fall*.

```model.py```: defines the skip-gram model, the loss function and its gradients
```data.py```: code for convering the HTML book into a token stream
```tokens.py```: code for building training contexts from the tokens
```training.py```: a simple SGD training routine

```training.ipynb``` holds an example of training code

For further discussion, see [here](briantimar.com/notes/daf)