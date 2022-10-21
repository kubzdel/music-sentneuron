from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from midi_encoder import write
from models import TransformerModel, TransformerLstmModel, LSTMModel, BigBirdLM, GPT2LM
from utils import top_k_top_p_filtering


class MidiTrainingModule(pl.LightningModule):
    """
    Training module with a Language Model head.
    Support Transformers as well as LSTMs.
    """

    def __init__(self, n_layer, n_head, n_embd, seq_len,
                 batch_size, epochs, samples_count, tokenizer, vocab_size, training_stride, validation_stride,ckpt=None) -> None:
        super(MidiTrainingModule, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.samples_count = samples_count
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        # build model
        # self.model = LSTMModel(vocab_size, 768, 4, 512)
        # self.model = TransformerLstmModel(n_layer=n_layer, n_head=n_head, n_embd=n_embd, seq_len=seq_len,
        #                               vocab_size=vocab_size, ckpt=None)
        self.model = GPT2LM(n_layer=n_layer, n_head=n_head, n_embd=n_embd, seq_len=seq_len,
                                      vocab_size=vocab_size)
        self.loss = torch.nn.CrossEntropyLoss()
        self.seq_len = seq_len
        self.save_hyperparameters(ignore='tokenizer')
    def generate(self, step) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """

        self.model.eval()
        # self.model.to('cpu')
        predicted_token = torch.Tensor([1])
        with torch.no_grad():
            output_seq = torch.as_tensor([2])
            while (
                    len(output_seq) < self.seq_len - 1 and predicted_token.unsqueeze(-1)[0] != 3
            ):
                if output_seq.ndim == 1:
                    outputs = self.forward(torch.unsqueeze(output_seq, 0), permute=False)
                else:
                    outputs = self.forward(output_seq, permute=False)
                lm_logits = torch.squeeze(outputs, 0)
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
                output_seq = torch.cat([output_seq.to('cpu'), predicted_token.to('cpu')])
            output_seq = [output_seq[1:-1].tolist()]
            output_sentence_midi = self.tokenizer.tokens_to_midi(output_seq)
            print(output_seq)
            Path(f'{self.run_name}_generated_samples').mkdir(exist_ok=True)
            midi_path = f'{self.run_name}_generated_samples/{self.run_name}_step_{str(step)}.mid'
            # write(output_sentence, midi_path)
            output_sentence_midi.dump(midi_path)
            self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=midi_path)
        self.model.cuda()
        self.model.train()
        return output_seq

    def on_fit_start(self):
        params = Namespace()
        self.logger.log_hyperparams(params)
        self.run_name = self.logger._run_name

    def forward(self, input_ids, labels=None, attention_mask=None, permute=True):
        if labels is not None:
            if permute:
                logits, loss = self.model(input_ids, labels, attention_mask)
                return logits.permute(0, 2, 1), loss
            else:
                return self.model(input_ids, labels, attention_mask)
        else:
            if permute:
                return self.model(input_ids, labels, attention_mask).permute(0, 2, 1)
            else:
                return self.model(input_ids, labels, attention_mask)

    def training_step(self, batch, batch_nb):
        # model_out = self.forward(batch['input_ids'])
        model_out, loss = self.forward(batch['input_ids'], batch['target_ids'], batch.get('attention_mask'))
        # loss = self.loss(model_out, batch['target_ids'])
        if batch_nb % 25 == 0:
            self.logger.log_metrics({'train_loss': loss}, step=batch_nb)
        return loss

    def validation_step(self, batch, batch_nb):
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        model_out, loss_val = self.forward(batch['input_ids'], batch['target_ids'], batch.get('attention_mask'))
        # loss_val = self.loss(model_out, batch['target_ids'])
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
                          lr=6e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8,
                          betas=(0.9, 0.999),
                          weight_decay=0.01
                          )

        t_total = int(self.samples_count / self.batch_size) * self.epochs
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_total)
        return [optimizer], [{"scheduler": self.sched, "interval": "step"}]

class MidiTrainingModuleNew(MidiTrainingModule):
    """
    Training module with a Language Model head.
    Support Transformers as well as LSTMs.
    """

    def __init__(self, n_layer, n_head, n_embd, seq_len,
                 batch_size, epochs, samples_count, tokenizer, vocab_size, training_stride, validation_stride,ckpt=None) -> None:
        super(MidiTrainingModule, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.samples_count = samples_count
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        # build model
        # self.model = LSTMModel(vocab_size, 768, 4, 512)
        self.model = TransformerLstmModel(n_layer=n_layer, n_head=n_head, n_embd=n_embd, seq_len=seq_len,
                                      vocab_size=vocab_size, ckpt=ckpt)
        self.loss = torch.nn.CrossEntropyLoss()
        self.seq_len = seq_len
        self.save_hyperparameters()
