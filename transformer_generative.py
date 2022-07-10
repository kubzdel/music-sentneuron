from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from midi_encoder import write
from models import TransformerModel
from utils import top_k_top_p_filtering


class MidiTrainingModule(pl.LightningModule):
    """
    Training module with a Language Model head.
    Support Transformers as well as LSTMs.
    """

    def __init__(self, n_layer, n_head, n_embd, seq_len,
                 batch_size, epochs, samples_count, tokenizer, vocab_size, training_stride, validation_stride) -> None:
        super(MidiTrainingModule, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.samples_count = samples_count
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        # build model
        # self.model = LSTMModel(vocab_size, embedding_size, lstm_layers, lstm_units)
        self.model = TransformerModel(n_layer=n_layer, n_head=n_head, n_embd=n_embd, seq_len=seq_len, vocab_size=vocab_size)
        self.loss = torch.nn.CrossEntropyLoss()
        self.seq_len = seq_len
        self.save_hyperparameters()

    def generate(self, step) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """

        self.model.eval()
        self.model.to('cpu')
        predicted_token = torch.Tensor([1])
        with torch.no_grad():
            output_seq = self.tokenizer.encode('\n', return_tensors=True)
            while (
                    len(output_seq) < self.seq_len-1 and predicted_token.unsqueeze(-1)[0] != 0
            ):
                outputs = self.forward(output_seq, permute=False)
                lm_logits = outputs
                logits = lm_logits[-1, :]
                top_k, top_p, temperature = 0, 0.95, 1
                filtered_logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p, temperature=temperature
                )
                probabilities = F.softmax(filtered_logits, dim=-1)
                probabilities_logits, probabilities_position = torch.sort(
                    probabilities, descending=True
                )
                predicted_token = torch.multinomial(probabilities, 1)
                output_seq = torch.cat([output_seq, predicted_token])
            output_seq = output_seq[1:-1]
            output_sentence = self.tokenizer.decode(output_seq)
            print(output_sentence)
            Path('val_generated').mkdir(exist_ok=True)
            midi_path = f'val_generated/generated_step_{str(step)}.mid'
            write(output_sentence, midi_path)
            self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=midi_path)
        self.model.cuda()
        self.model.train()
        return output_sentence

    def on_fit_start(self):
        params = Namespace()
        self.logger.log_hyperparams(params)

    def forward(self, input_ids, permute=True):
        if permute:
            return self.model(input_ids).permute(0, 2, 1)
        else:
            return self.model(input_ids)

    def training_step(self, batch, batch_nb):
        model_out = self.forward(batch['input_ids'])
        loss = self.loss(model_out, batch['target_ids'])
        if batch_nb % 25 == 0:
            self.logger.log_metrics({'train_loss': loss}, step=batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        model_out = self.forward(batch['input_ids'])
        loss_val = self.loss(model_out, batch['target_ids'])

        output = OrderedDict({"val_loss": loss_val})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        """
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            # avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
            # print(self.metrics_manager.valid_results_to_string())
            self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
            self.logger.log_metrics({'val_loss': avg_loss}, step=self.global_step)
        self.generate(self.global_step)

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
