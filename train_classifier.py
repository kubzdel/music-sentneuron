import os
import csv
import json
import pickle
import argparse
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import PreTrainedTokenizerFast

import midi_encoder as me
import plot_results as pr
from tokenizer import MidiTokenizer

from train_generative import build_generative_model, build_char2idx
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier

# Directory where trained model will be saved
from transformer_generative import MidiTrainingModule

TRAIN_DIR = "./trained"

def preprocess_sentence(text, front_pad='\n ', end_pad=''):
    text = text.replace('\n', ' ').strip()
    return '\n ' + text + ' \n'

def encode_sentence(model, text, tokenizer, layer_idx):
    text = preprocess_sentence(text)

    # Reset LSTMs hidden and cell states
    # model.reset_states()
    model.eval()
    with torch.no_grad():

            # Add the batch dimension
        input_eval = torch.tensor(tokenizer.encode(text), dtype=torch.int)
        input_eval = torch.unsqueeze(input_eval, 0)
        input_eval = input_eval.to('cuda')
        gpt2_outputs = model.gpt2(input_eval)
        hidden_states = gpt2_outputs.last_hidden_state
        linear_state = model.linear_layer(hidden_states)
        # last_token_embedding = torch.cat((hidden_states[:, -1, :480], hidden_states[:, 0, 480:]), dim=-1)
        last_token_embedding = linear_state[:,-1,:]
        last_token_embedding_squeezed = torch.squeeze(last_token_embedding, 0)
        # sss = torch.mean(torch.squeeze(linear_state,0), dim=0)
        # token_mean_embedding2 = torch.max_pool1d(torch.squeeze(linear_state, 0), len(last_token_embedding_squeezed))
        token_mean_embedding = torch.mean(torch.squeeze(linear_state, 0), 0)
        # remove the batch dimension
        #h_state = tf.squeeze(h_state, 0)
        # c_state = torch.squeeze(last_token_embedding, 0)
        return token_mean_embedding.cpu().detach().numpy()
    # return mmm.token_mean_embedding().numpy()

def build_dataset(datapath, generative_model, tokenizer, layer_idx):
    xs, ys = [], []

    csv_file = open(datapath, "r")
    data = csv.DictReader(csv_file)

    for row in data:
        label = int(row["label"])
        filepath = row["filepath"]

        data_dir = os.path.dirname(datapath)
        phrase_path = os.path.join(data_dir, filepath) + ".mid"
        encoded_path = os.path.join(data_dir, filepath) + ".npy"

        # Load midi file as text
        if os.path.isfile(encoded_path):
            encoding = np.load(encoded_path)
        else:
            text, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)

            # Encode midi text using generative lstm
            try:
                with torch.no_grad():
                    if text:
                        encoding = encode_sentence(generative_model, text, tokenizer, layer_idx)
            except Exception as e:
                continue

            # Save encoding in file to make it faster to load next time
            # np.save(encoded_path, encoding)

        xs.append(encoding)
        ys.append(label)

    return np.array(xs), np.array(ys)

def train_classifier_model(train_dataset, test_dataset, C=2**np.arange(-8, 1).astype(np.float), seed=42, penalty="l1"):
    trX, trY = train_dataset
    teX, teY = test_dataset

    scores = []

    # Hyper-parameter optimization
    for i, c in enumerate(C):
        logreg_model = LogisticRegression(max_iter=1000,tol=1e-8, C=c, penalty=penalty, random_state=seed+i, solver="liblinear")

        logreg_model.fit(trX, trY)

        score = logreg_model.score(teX, teY)
        scores.append(score)

    c = C[np.argmax(scores)]

    sent_classfier = LogisticRegression(max_iter=1000,tol=1e-8, C=c, penalty=penalty, random_state=seed+len(C), solver="liblinear")
    sent_classfier.fit(trX, trY)

    score =  sent_classfier.score(teX, teY) * 100.
    from sklearn.metrics import f1_score
    print(score)

    # Persist sentiment classifier
    with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
        pickle.dump(sent_classfier, f)

    # Get activated neurons
    sentneuron_ixs = get_activated_neurons(sent_classfier)

    # Plot results
    pr.plot_weight_contribs(sent_classfier.coef_)
    pr.plot_logits(trX, trY, sentneuron_ixs)

    return sentneuron_ixs, score

def get_activated_neurons(sent_classfier):
    neurons_not_zero = len(np.argwhere(sent_classfier.coef_))
    np.where(sent_classfier.coef_)
    weightss = np.where(sent_classfier.coef_ > abs(0.1))
    weights = sent_classfier.coef_.T
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

    if neurons_not_zero == 1:
        neuron_ixs = np.array([np.argmax(weight_penalties)])
    elif neurons_not_zero >= np.log(len(weight_penalties)):
        neuron_ixs = np.argsort(weight_penalties)[-neurons_not_zero:][::-1]
    else:
        neuron_ixs = np.argpartition(weight_penalties, -neurons_not_zero)[-neurons_not_zero:]
        neuron_ixs = (neuron_ixs[np.argsort(weight_penalties[neuron_ixs])])[::-1]

    return neuron_ixs
# [ 13  23  24  46  51  60  73  83  84  90 102 105 108 124 139 140 150 153, 168 173 176 180 191 194 201 220 236 241 249 253 278 282 297 308 330 333, 348 361 374 385 386 388 389]
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_classifier.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test' , type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=False, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--cellix', type=int, required=True, help="LSTM layer to use as encoder.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="new_tokenizer_word_big.json")
    tokenizer.unk_token = "<UNK>"
    vocab_size = len(tokenizer)
    # Rebuild generative model from checkpoint
    generative_model = MidiTrainingModule.load_from_checkpoint('gpt2_new_dataset/last.ckpt', batch_size=4,
                                                               epochs=2, samples_count = 100, tokenizer = None, embedding_size = 768,
                                                               vocab_size = vocab_size, lstm_layers = 3, lstm_units = 3, strict=False,
                                                               n_layer=4, n_head = 4, n_embd = 768, seq_len=512, training_stride=512, validation_stride=512).model
    generative_model.eval()
    generative_model.to('cuda')
    # Build dataset from encoded labelled midis
    train_dataset = build_dataset(opt.train, generative_model, tokenizer, opt.cellix)
    test_dataset = build_dataset(opt.test, generative_model, tokenizer, opt.cellix)

    # Train model
    sentneuron_ixs, score = train_classifier_model(train_dataset, test_dataset)

    print("Total Neurons Used:", len(sentneuron_ixs), "\n", sentneuron_ixs)
    print("Test Accuracy:", score)
