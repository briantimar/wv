import numpy as np
from datetime import datetime
import json
import time

def sgd_train(model, context_iter, epochs, lr, wt_decay=0.0, logstep=100, verbose=False, logfile=None):
    """ Train the model via SGD on the given dataset. Gradients are *not* batched across different contexts.
        model: a SkipGramVW model.
        context_iter: iterator over input words and the corresponding context and noise tokens.
        epochs: how many passes to take over the full dataset.
        lr: learning rate
        wt_decay: weight decay coefficient (added to the loss function).
        logstep: number of updates between log steps
        verbose: whether to print stuff to terminal
        logfile: where to write logging data."""
    
    losses = []
    smoothed_losses = []

    if logfile is None:
        logfile = datetime.now().strftime("sgd_log_%y_%m_%d__%H_%M_%S.json")

    def log():
        _log = {'loss': smoothed_losses, 'logstep': logstep, 
                'lr': lr, 'wt_decay': wt_decay, 'epochs': epochs}
        with open(logfile, 'w') as f:
            json.dump(_log, f)

    for ep in range(epochs):
        t0 = time.time()
        for i, (input_index, context, noise) in enumerate(context_iter):
            tb = time.time()
            model.do_sgd_update(input_index, context, noise, lr, wt_decay=wt_decay)
            tf = time.time()
            tload = tb - t0
            tsgd = tf - tb
            losses.append(model.neg_loss(input_index, context, noise))
            tupdate = tload + tsgd
            if i % logstep == 0:
                smoothed_losses.append(np.mean(losses))
                losses = []
                if verbose:
                    print(f"Word {i}: update time {tupdate} sec, loss {smoothed_losses[-1]}")
                log()
            t0 = time.time()
    return times, losses

if __name__ == "__main__":
    from tokens import TokenSet, ContextIterator
    from model import SkipGramWV
    tokenfile = "data/gibbon_daf_tokens.txt"
    ts = TokenSet(tokenfile)
    ci = ContextIterator(ts, 5, num_noise=10)
    model = SkipGramWV(ts.num_tokens, 10)
    sgd_train(model, ci, 1, .1, logstep=1000, logfile="log.json", verbose=True)
