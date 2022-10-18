from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
from prettytable import PrettyTable
from models import TransformerClassificationModel


class MidiClassificationModule(pl.LightningModule):
    """
    Training module with a Language Model head.
    Support Transformers as well as LSTMs.
    """

    def __init__(self, model,
                 batch_size, epochs, samples_count) -> None:
        super(MidiClassificationModule, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.samples_count = samples_count
        self.model = model
        self.model = TransformerClassificationModel(model=model)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(num_classes=2)
        self.f1 = F1Score(num_classes=2, average='none').to('cpu')
        self.save_hyperparameters()

    def on_fit_start(self):
        # params = Namespace()
        # self.logger.log_hyperparams(params)
        self.run_name = "new_tokenizer"

    def forward(self, input_ids, permute=True):
            return self.model(input_ids)

    def training_step(self, batch, batch_nb):
        model_out = self.forward(batch['encoding'])
        loss = self.loss(model_out, batch['class'].to(torch.float32))
        if batch_nb % 25 == 0:
            self.logger.log_metrics({'train_loss': loss}, step=batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        model_out = self.forward(batch['encoding'])
        loss_val = self.loss(model_out, batch['class'].to(torch.float32))

        logits = model_out.detach().cpu()
        logits = torch.sigmoid(logits.to(torch.float))
        label_ids = batch['class'].to('cpu')

        accuracy = self.accuracy(logits, label_ids)
        self.f1.to('cpu')
        f1 = self.f1(logits.to('cpu'), label_ids)

        output = OrderedDict({"val_loss": loss_val, "accuracy": accuracy, "f1":f1})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        """
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
            avg_f1 = torch.stack([x['f1'] for x in outputs]).nanmean(dim=0)
            # avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
            # print(self.metrics_manager.valid_results_to_string())
            self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
            self.log('accuracy', avg_acc, on_epoch=True, prog_bar=True)
            self.logger.log_metrics({'val_loss': avg_loss}, step=self.global_step)
            self.logger.log_metrics({'accuracy': avg_acc}, step=self.global_step)
            self.print_f1(avg_f1)
            print(avg_loss)

    def print_f1(self, f1_results):
        mean = f1_results.nanmean().item()
        f1_results = f1_results.tolist()
        x = PrettyTable()
        x.field_names = ['Category'] + ['F1']

        for label, score in zip(['positive','negative'], f1_results):
            x.add_row([label] + [score])
        x.add_row(['Mean'] + [mean])
        print(x.get_string())
    def configure_optimizers(self):
        """ Sets Learning rate for different parameter groups. """
        optimizer = AdamW(self.model.parameters(),
                          lr=3e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8,
                          betas=(0.9, 0.999),
                          weight_decay=0.01
                          )

        t_total = int(self.samples_count / self.batch_size) * self.epochs
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_total)
        return [optimizer], [{"scheduler": self.sched, "interval": "step"}]
