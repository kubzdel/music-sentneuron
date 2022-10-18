import argparse
import os
from pathlib import Path

import pytorch_lightning
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from transformers import PreTrainedTokenizerFast

import midi_encoder as me
from data_module import MusicDataModule
from train_generative import build_char2idx
from transformer_generative import MidiTrainingModule

pytorch_lightning.seed_everything(0)
load_dotenv()
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_generative.py')
    parser.add_argument('--run_name', type=str, required=True, help="MLFlow run name")
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test', type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=False, help="Checkpoint dir.")
    parser.add_argument('--n_embd', type=int, default=256, help="Embedding size.")
    parser.add_argument('--n_head', type=int, default=512, help="Transformer heads.")
    parser.add_argument('--n_layer', type=int, default=2, help="Layers.")
    parser.add_argument('--batch', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs.")
    parser.add_argument('--seq_len', type=int, default=100, help="Sequence length.")
    parser.add_argument('--training_stride', type=int, default=50, help="Stride length.")
    parser.add_argument('--validation_stride', type=int, default=50, help="Stride length.")
    opt = parser.parse_args()

    # Encode midi files as text with vocab
    train_text, train_vocab = me.load(opt.train, ignore=True)
    test_text, test_vocab = me.load(opt.test, ignore=True)

    # Build dictionary to map from char to integers
    char2idx, vocab_size = build_char2idx(train_vocab, test_vocab)

    result_path = os.path.join('', opt.run_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    # tokenizer = MidiTokenizer(from_file='trained/char2idx.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="new_tokenizer_word_old.json")
    tokenizer.unk_token = "<UNK>"
    tokenizer.pad_token = "<PAD>"
    tokenizer.bos_token = "\n"
    tokenizer.eos_token = "\n"
    vocab_size = len(tokenizer)
    data = MusicDataModule(train_input=train_text, val_input=test_text, tokenizer=tokenizer,
                           sequence_length=opt.seq_len, train_stride=opt.training_stride,
                           val_stride=opt.validation_stride,
                           num_workers=8, batch_size=opt.batch)
    # data.setup()
    # training_samples = len(data.train_dataset)
    training_samples = len(train_text.split('\n'))
    # ckpt = MidiTrainingModule.load_from_checkpoint('trained_model/epoch=1-step=10571.ckpt',
    #                                         batch_size=4,
    #                                         epochs=2, samples_count=100, tokenizer=None,
    #                                         embedding_size=10,
    #                                         vocab_size=vocab_size, lstm_layers=3, lstm_units=3,
    #                                         strict=False).model
    model = MidiTrainingModule(batch_size=opt.batch, epochs=opt.epochs, samples_count=training_samples,
                               tokenizer=tokenizer, n_embd=opt.n_embd, n_layer=opt.n_layer,
                               n_head=opt.n_head, vocab_size=vocab_size,
                               seq_len=opt.seq_len, training_stride=opt.training_stride,
                               validation_stride=opt.validation_stride)
    checkpoint_callback = ModelCheckpoint(
        dirpath=result_path,
        save_top_k=5,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        every_n_epochs=1,
        mode="min")

    mlf_logger = MLFlowLogger(experiment_name="generative-midi", tracking_uri="http://150.254.131.193:5000/",
                              run_name=opt.run_name)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    load_dotenv()
    trainer = Trainer(gpus=1, max_epochs=opt.epochs, callbacks=[checkpoint_callback],
                      logger=mlf_logger,
                      log_every_n_steps=20, val_check_interval=0.25, progress_bar_refresh_rate=20, precision=16,
                      accumulate_grad_batches=1)
    # lr_finder = trainer.tuner.lr_find(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader(),min_lr=1e-7, max_lr=1e-2,num_training=1000)
    #
    # # Results can be found in
    # print(lr_finder.results)
    #
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(new_lr)

    # update hparams of the mod
    trainer.fit(model, data)
