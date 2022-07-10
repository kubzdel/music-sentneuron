import argparse
import os
from pathlib import Path

import pytorch_lightning
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

import midi_encoder as me
from data_module import MusicDataModule
from tokenizer import MidiTokenizer
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
    train_text, train_vocab = me.load(opt.train)
    test_text, test_vocab = me.load(opt.test)

    # Build dictionary to map from char to integers
    char2idx, vocab_size = build_char2idx(train_vocab, test_vocab)

    result_path = os.path.join('', 'trained_model')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    tokenizer = MidiTokenizer(from_file='trained/char2idx.json')
    data = MusicDataModule(train_input=train_text, val_input=test_text, tokenizer=tokenizer,
                           sequence_length=opt.seq_len, train_stride=opt.training_stride,
                           val_stride=opt.validation_stride,
                           num_workers=0, batch_size=opt.batch)
    data.setup()
    training_samples = len(data.train_dataset)
    model = MidiTrainingModule(batch_size=opt.batch, epochs=opt.epochs, samples_count=training_samples,
                               tokenizer=tokenizer, n_embd=opt.n_embd, n_layer=opt.n_layer,
                               n_head=opt.n_head, vocab_size=vocab_size,
                               seq_len=opt.seq_len, training_stride=opt.training_stride,
                               validation_stride=opt.validation_stride)
    checkpoint_callback = ModelCheckpoint(
        dirpath=result_path,
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        every_n_epochs=1,
        mode="min")

    mlf_logger = MLFlowLogger(experiment_name="generative-midi", tracking_uri="http://150.254.131.193:5000/",
                              run_name=opt.run_name)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    load_dotenv()
    trainer = Trainer(gpus=1, max_epochs=opt.epochs, callbacks=[early_stop_callback, checkpoint_callback],
                      logger=mlf_logger,
                      log_every_n_steps=20, val_check_interval=0.2, progress_bar_refresh_rate=20, precision=16,
                      accumulate_grad_batches=1, limit_val_batches=2000)
    trainer.fit(model, data)
