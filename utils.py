from time import gmtime, strftime, time
import numpy as np
import torch
import os


def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print ("{} - {} sec".format(method.__name__, te-ts))
        return result
    return timed


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class EarlyStopping:
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, epoch=-1, verbose=False, delta=0, trace_func=print):
    #def __init__(self, patience=7, epoch=0, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.epoch = epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model, model_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print("EarlyStopping counter is higher than patience")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_name):
        '''Saves model when validation loss decrease.'''
        if not os.path.isdir('./checkpoints/'):
            os.makedirs('./checkpoints/')
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        path = 'checkpoints/' + model_name + 'checkpoint_{}.ckp'.format(get_datetime())
        torch.save({'state_dict': model.state_dict()}, path)
        self.val_loss_min = val_loss
        #t.save({'state_dict': self._model.state_dict()}, 'checkpoints/' + model_name + 'checkpoint.ckp')


def get_datetime():
    #return strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    # storing per day to have different runs from different days
    return strftime("%Y-%m-%d", gmtime())

"""
def log(log_message, verbose=False):
    if verbose:
        print("[{}]{}".format(get_datetime(), log_message))


def log_success(log_message, verbose=True):
    log("\033[32m{}\033[0m".format(log_message), verbose)


def log_major(log_message, verbose=True):
    log("\033[1m{}\033[0m".format(log_message), verbose)


def set_default_tensor():
    if torch.cuda.is_available():
        print("Using GPU. Setting default tensor type to torch.cuda.FloatTensor")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("Using CPU. Setting default tensor type to torch.FloatTensor")
        torch.set_default_tensor_type("torch.FloatTensor")
"""
