import torch

from tqdm import tqdm

from .utils.torch import normalize_embeddings
from .utils.torch import to_var


class Trainer(object):
    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        num_epochs=10,
        use_cuda=True,
        log_interval=20,
    ):
        self.model = model

        self.optimizer = optimizer
        self.loss_function = loss_function

        self.num_epochs = num_epochs

        self.use_cuda = use_cuda
        self.log_interval = log_interval

    def train_epoch(
        self, train_batches, epoch, silent=False, embeddings_norm_dim=None
    ):
        self.model.train()  # Depends on using pytorch
        num_batches = train_batches.num_batches

        total_loss = 0
        for batch_index in tqdm(range(num_batches), desc="Batch", disable=silent):
            self.model.zero_grad()
            batch = train_batches[batch_index]
            ret_dict = self.model(batch)

            # FIXME: This part depends both on the way the batch is built and
            # on using pytorch. Think of how to avoid this. Maybe by creating
            # a specific MultNLI Trainer Subclass?
            labels = batch["labels"]
            labels = to_var(
                torch.LongTensor(labels), self.use_cuda, requires_grad=False
            )

            batch_loss = self.loss_function(ret_dict["logits"], labels)
            batch_loss.backward()

            self.optimizer.step()

            if embeddings_norm_dim is not None:
                normalize_embeddings(
                    self.model.embeddings, dim=embeddings_norm_dim, order=2
                )
                normalize_embeddings(
                    self.model.char_embeddings, dim=embeddings_norm_dim, order=2
                )

            # We ignore batch 0's output for prettier logging
            if batch_index != 0:
                total_loss += batch_loss.item()

            if (
                batch_index % self.log_interval == 0
                and batch_index != 0
                and not silent
            ):

                avg_loss = total_loss / self.log_interval
                tqdm.write(
                    f"Epoch: {epoch}, batch: {batch_index}, loss: {avg_loss}"
                )
                total_loss = 0
