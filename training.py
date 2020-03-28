import numpy as np

def sgd_train(model, context_iter, epochs, lr, wt_decay=0.0, logstep=100):
    """ Train the model via SGD on the given dataset. Gradients are *not* batched across different contexts.
        model: a SkipGramVW model.
        context_iter: iterator over input words and the corresponding context and noise tokens.
        epochs: how many passes to take over the full dataset.
        lr: learning rate
        wt_decay: weight decay coefficient (added to the loss function)."""
    load_times = []
    sgd_times = []
    losses = []
    for ep in range(epochs):
        t0 = time.time()
        for i, (input_index, context, noise) in enumerate(context_iter):
            tb = time.time()
            model.do_sgd_update(input_index, context, noise, lr, wt_decay=wt_decay)
            tf = time.time()
            tload = tb - t0
            tsgd = tf - tb
            tupdate = tload + tsgd
            if i % logstep == 0:
                load_times.append(tload)
                sgd_times.append(tsgd)
                loss = model.neg_loss(input_index, context, noise)
                print(f"Word {i}: update time {tupdate} sec, loss {loss}")
            t0 = time.time()
    return times, losses
