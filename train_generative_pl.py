import argparse
import os
from pathlib import Path

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

pytorch_lightning.seed_everything(0)
import midi_encoder as me
from data_module import MusicDataModule
from tokenizer import MidiTokenizer
from train_generative import build_char2idx
from transformer_generative import MidiTrainingModule

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_generative.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test', type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=False, help="Checkpoint dir.")
    parser.add_argument('--embed', type=int, default=256, help="Embedding size.")
    parser.add_argument('--units', type=int, default=512, help="LSTM units.")
    parser.add_argument('--layers', type=int, default=2, help="LSTM layers.")
    parser.add_argument('--batch', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs.")
    parser.add_argument('--seqlen', type=int, default=100, help="Sequence lenght.")
    parser.add_argument('--stride', type=int, default=50, help="Stride lenght.")
    parser.add_argument('--drop', type=float, default=0.0, help="Dropout.")
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
                           sequence_length=opt.seqlen, stride=opt.stride, num_workers=0, batch_size=opt.batch)
    data.setup()
    training_samples = len(data.train_dataset)
    model = MidiTrainingModule(batch_size=opt.batch, epochs=opt.epochs, samples_count=training_samples,
                               tokenizer=tokenizer, embedding_size=opt.seqlen, vocab_size=vocab_size,
                               lstm_layers=opt.layers,
                               lstm_units=opt.units)
    checkpoint_callback = ModelCheckpoint(
        dirpath=result_path,
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        every_n_epochs=1,
        mode="min")

    trainer = Trainer(gpus=1, max_epochs=opt.epochs, callbacks=[checkpoint_callback],
                      log_every_n_steps=5, val_check_interval=1200, progress_bar_refresh_rate=5, precision=32,
                      accumulate_grad_batches=1, limit_val_batches=800)

    trainer.fit(model, data)
