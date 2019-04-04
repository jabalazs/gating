"""
Taken from
https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Optim.py
"""

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class ScheduledOptim(object):

    """Optimizer with predefined learning rate schedule"""

    def __init__(self, core_optimizer, lr_scheduler):
        """

        Parameters
        ----------
        core_optimizer : torch.optim.Optimizer
        lr_scheduler : Scheduler instance


        """
        self.optimizer = core_optimizer
        self.lr_scheduler = lr_scheduler

        self.step_num = 0
        self.lr = None

    def step(self):
        self.step_num += 1
        rate = self.get_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.lr = rate
        self.optimizer.step()

    def get_rate(self, step=None):
        if step is None:
            step = self.step_num

        return self.lr_scheduler.get_rate(step)

    def zero_grad(self):
        self.optimizer.zero_grad()


class OptimWithDecay(object):
    def _makeOptimizer(self):
        if self.method == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == "adagrad":
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == "adadelta":
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == "adam":
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        elif self.method == "rmsprop":
            self.optimizer = optim.RMSprop(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, initial_lr, max_grad_norm=None):

        self.params = list(params)  # careful: params may be a generator
        self._filter_params()
        self.last_ppl = None
        self.lr = initial_lr
        self.max_grad_norm = max_grad_norm
        self.method = method

        self._makeOptimizer()

        self.last_accuracy = 0

    def _filter_params(self):
        self.params = [param for param in self.params if param.requires_grad]

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def updt_lr_accuracy(self, epoch, accuracy):
        """This is the lr update policy that Conneau used for Infersent in
           https://arxiv.org/abs/1705.02364"""
        updated = False
        if accuracy < self.last_accuracy:
            self.lr = self.lr / 5
            updated = True

        self.last_accuracy = accuracy

        self._makeOptimizer()
        return updated, self.lr
