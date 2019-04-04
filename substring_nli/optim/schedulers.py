import numpy as np


class SlantedTriangularScheduler(object):

    """Scheduler producing a slanted triangular learning rate schedule

       From Howard & Ruder's (2018) paper:
       Universal Language Model Fine-tuning for Text Classification
       https://arxiv.org/abs/1801.06146
    """

    def __init__(self, max_step, max_lr, cut_fraction, ratio):
        """

        Parameters
        ----------
        max_step : int
            Last training step (probably should equal num_batches * num_epochs)
        max_lr : float, optional
            Maximum desired learning rate
        cut_fraction : int, optional
            Fraction of steps during which to increase the learning rate
        ratio : int, optional
            How many times bigger is the maximum learning rate as compared to the
            minimum one

        """
        self.max_step = max_step
        self.max_lr = max_lr
        self.cut_fraction = cut_fraction
        self.ratio = ratio

    def get_rate(self, step):
        """
        Parameters
        ----------
            step : int
                Current step during training

        Returns
        -------
        learning_rate : float
            The learning rate for a given step
        """
        # FIXME: Line below has not been tested
        cut = np.floor(self.max_step * self.cut_fraction)

        if step < cut:
            p = step / cut
        else:
            p = 1 - ((step - cut) / (cut * (1 / self.cut_fraction - 1)))
        learning_rate = self.max_lr * (1 + p * (self.ratio - 1)) / self.ratio

        return learning_rate


class TransformerScheduler(object):

    """Implement linear lr growth and O(sqrt(n)) lr decay

    From Attention is All You Need (2017)
    https://arxiv.org/abs/1706.03762"""

    def __init__(self, model_size, factor, warmup_steps):
        """

        Parameters
        ----------
        model_size : int
            Dimensionality of vector returned by encoder
        factor : float
            Multiplicative factor for lr returned by scheduler
        warmup_steps : int
            During how many steps to increase the learning rate


        Returns
        -------
        learning_rate : float
            The learning rate corresponding to the current step

        """
        self.model_size = model_size
        self.factor = factor
        self.warmup_steps = warmup_steps

    def get_rate(self, step):
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )


class DANNScheduler(object):

    """Docstring for DANNScheduler. """

    def __init__(self, max_step, mu_0=1e-2, alpha=10, beta=0.75):
        """
        Parameters
        ----------
        max_step : int
        mu_0 : float, optional
        alpha : float, optional
        beta : float, optional


        """
        self.max_step = max_step

        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta

    def get_rate(self, step):
        p = step / self.max_step
        return self.mu_0 / np.power(1 + self.alpha * p, self.beta)
