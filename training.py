import numpy as np
from datetime import datetime
import json
import time
import os

def sgd_train(model, context_iter, epochs, lr_fn,
                model_dir, wt_decay=0.0, logstep=100, verbose=False, logfile=None, 
                modelstep=200000, 
                log_params = {}):
    """ Train the model via SGD on the given dataset. Gradients are *not* batched across different contexts.
        model: a SkipGramVW model.
        context_iter: iterator over input words and the corresponding context and noise tokens.
        epochs: how many passes to take over the full dataset.
        lr_fn: lambda called at each word to produce learning rate
        wt_decay: weight decay coefficient (added to the loss function).
        logstep: number of updates between log steps
        verbose: whether to print stuff to terminal
        logfile: where to write logging data."""
    
    losses = []
    smoothed_losses = []

    if logfile is None:
        logfile = datetime.now().strftime("sgd_log_%y_%m_%d__%H_%M_%S.json")
    log_params.update({'loss': smoothed_losses, 'logstep': logstep, 
                 'wt_decay': wt_decay, 'epochs': epochs})
    def log():
        with open(logfile, 'w') as f:
            json.dump(log_params, f)
    try:
        log_params["t_start"] = str(datetime.now())
        for ep in range(epochs):
            t0 = time.time()
            for i, (input_index, context, noise) in enumerate(context_iter):
                tb = time.time()
                model.do_sgd_update(input_index, context, noise, lr_fn(), wt_decay=wt_decay)
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
                if i % modelstep == 0:
                    np.save(os.path.join(model_dir, f"model_{i//modelstep}"), model._weights)
                t0 = time.time()
    except KeyboardInterrupt:
        print("Halting early")
    log_params["t_final"] = str(datetime.now())
    log()
    np.save(os.path.join(model_dir, "model_final"), model._weights)
    return model, smoothed_losses

if __name__ == "__main__":
    pass

    # from tokens import TokenSet, ContextIterator
    # from model import SkipGramWV
    # tokenfile = "data/gibbon_daf_tokens.txt"
    # ts = TokenSet(tokenfile)
    # context_radius = 5
    # num_noise = 100
    # vector_dim = 10
    # epochs = 1
    # lr = .01
    # logstep = 1000
    # ci = ContextIterator(ts, context_radius, num_noise=num_noise)
    # model = SkipGramWV(ts.num_tokens, vector_dim)
    # log_params = dict(num_noise=num_noise, vector_dim=vector_dim,context_radius=context_radius)
    # sgd_train(model, ci, epochs, lr, logstep=logstep, logfile="log.json", 
    #                 verbose=True, log_params=log_params)
