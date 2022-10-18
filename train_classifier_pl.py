import argparse
import json
import os
from pathlib import Path

import pytorch_lightning
import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from transformers import PreTrainedTokenizerFast

import midi_encoder as me
from data_module import MusicDataModule
from symbolic_dataset import ClassificationDataset
from tokenizer import MidiTokenizer
from train_generative import build_char2idx
from transformer_classifier import MidiClassificationModule
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
    parser.add_argument('--batch', type=int, default=64, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    opt = parser.parse_args()
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    result_path = os.path.join('', opt.run_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="new_tokenizer_word_sturm.json")
    tokenizer.unk_token = "<UNK>"
    tokenizer.pad_token = "<PAD>"
    tokenizer.bos_token = "\n"
    tokenizer.eos_token = "\n"
    vocab_size = len(tokenizer)
    generative_model = MidiTrainingModule.load_from_checkpoint('gpt2_r1/last.ckpt', batch_size=4,
                                                               epochs=2, samples_count = 100, tokenizer = None, embedding_size = 10,
                                                               vocab_size = vocab_size, lstm_layers = 3, lstm_units = 3,
                                                               n_layer=6, n_head = 6, n_embd = 300, seq_len=1200, training_stride=1200, validation_stride=1200).model.gpt2
    generative_model.save_pretrained('gpt2_model_sturm_new')
    generative_model = 'gpt2_model_sturm_new'
    train_dataset = ClassificationDataset(opt.train, tokenizer, 400)
    test_dataset = ClassificationDataset(opt.test, tokenizer, 400)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch, shuffle=False)

    training_samples = len(train_dataset)
    model = MidiClassificationModule(batch_size=opt.batch, epochs=opt.epochs, samples_count=training_samples,
                                     model=generative_model)
    checkpoint_callback = ModelCheckpoint(
        dirpath=result_path,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        every_n_epochs=1,
        mode="min")

    mlf_logger = MLFlowLogger(experiment_name="classification-midi", tracking_uri="http://150.254.131.193:5000/",
                              run_name=opt.run_name)
    early_stop_callback = EarlyStopping(monitor="accuracy", min_delta=0.00, patience=5, verbose=False, mode="max")
    load_dotenv()
    trainer = Trainer(gpus=1, max_epochs=opt.epochs, callbacks=[checkpoint_callback],
                      logger=mlf_logger,
                      log_every_n_steps=20, val_check_interval=0.25, progress_bar_refresh_rate=20, precision=16,
                      accumulate_grad_batches=1, limit_val_batches=2000)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
