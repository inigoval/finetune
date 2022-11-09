import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F

from einops import rearrange
from typing import Any, Dict, List, Tuple, Type

from torch import Tensor
from networks.models import LogisticRegression


class FineTune(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder,
        dim,
        n_classes,
        n_epochs=100,
        n_layers=0,
        batch_size=1024,
        lr_decay=0.75,
        seed=69,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.freeze = True if n_layers == 0 else False
        self.batch_size = batch_size
        self.encoder = encoder
        self.lr_decay = lr_decay
        # TODO add MLP head choice
        self.head = LogisticRegression(input_dim=dim, output_dim=n_classes)
        self.n_epochs = n_epochs
        self.seed = seed

        self.val_acc = tm.Accuracy(average="micro", threshold=0)
        self.test_acc = tm.Accuracy(average="micro", threshold=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):

        # Log size of data-sets #
        logging_params = {key: len(value) for key, value in self.trainer.datamodule.data.items()}
        self.logger.log_hyperparams(logging_params)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.freeze else 0)
        self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.val_acc(preds, y)
        self.log(f"finetuning/val_acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)
        self.test_acc(preds, y)
        self.log(f"finetuning/test_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.freeze:
            # Scale base lr=0.1
            lr = 0.1 * self.batch_size / 256
            params = self.head.parameters()
            return torch.optim.SGD(params, momentum=0.9, lr=lr)
        else:
            lr = 0.001 * self.batch_size / 256
            params = [{"params": self.head.parameters(), "lr": lr}]
            layers = self.encoder.finetuning_layers[::-1]
            # layers.reverse()
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            # Append parameters of layers for finetuning along with decayed learning rate
            for i, layer in enumerate(layers[: self.n_layers]):
                params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

            # Initialize AdamW optimizer with cosine decay learning rate
            opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
            return [opt], [scheduler]


def run_finetuning(config, encoder, datamodule, logger):

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        # dirpath=config["files"] / config["run_id"] / "finetuning",
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        filename="{epoch}",  # filename may not work here TODO
        save_weights_only=True,
        # save_top_k=3,
    )

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint],
        max_epochs=config["finetune"]["n_epochs"],
        **config["trainer"],
    )

    model = FineTune(encoder, **config["finetune"])

    trainer.fit(model, datamodule)

    trainer.test(model, dataloaders=datamodule)

    return checkpoint, model
