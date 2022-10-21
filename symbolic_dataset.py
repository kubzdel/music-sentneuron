import csv
import itertools
import os
from collections import Counter
from multiprocessing.pool import ThreadPool, Pool

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast

import midi_encoder as me
from tokenizer import MidiTokenizer


class SymbolicDataset(Dataset):
    def __init__(self, symbolic_text, tokenizer, sequence_length=256, stride=100):
        self.tokenizer = tokenizer
        self.stride = stride
        self.sequence_length = sequence_length
        self.x_input, self.y_input = self.build_dataset(symbolic_text, tokenizer, sequence_length, stride)

    def pytorch_rolling_window(self, x, window_size, step_size=1):
        # unfold dimension to make our rolling window
        return x.unfold(0, window_size, step_size)

    def build_dataset(self, text, tokenizer : PreTrainedTokenizerFast, sequence_length, stride):
        list_of_ints = [tokenizer.encode(t.strip() + " \n") for t in text.split('\n')]
        # text_as_int = np.array([elem for sublist in list_of_ints for elem in sublist])
        text_as_int = np.array(list(itertools.chain.from_iterable(list_of_ints)))
        x_tensor = self.pytorch_rolling_window(torch.from_numpy(text_as_int[:-1]), sequence_length, stride)
        y_tensor = self.pytorch_rolling_window(torch.from_numpy(text_as_int[1:]), sequence_length, stride)
        return x_tensor, y_tensor

    def __len__(self):
        return self.x_input.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.x_input[idx], 'target_ids': self.y_input[idx].type(torch.LongTensor)}

class RemiDatasetTruncate(Dataset):
    def __init__(self, symbolic_text, tokenizer, sequence_length=256, stride=100):
        self.tokenizer = tokenizer
        self.stride = stride
        self.sequence_length = sequence_length
        self.x_input, self.y_input = self.build_dataset(symbolic_text)

    def pytorch_rolling_window(self, x, window_size, step_size=1):
        # unfold dimension to make our rolling window
        return x.unfold(0, window_size, step_size)

    def encode_single_piece(self, t):
        piece = t.strip().split()
        piece.append('\n')
        piece.insert(0, '\n')
        max_len = max(len(piece), self.sequence_length)
        tokenized_chunk = self.tokenizer.encode_plus(piece[:max_len], max_length=self.sequence_length, padding='max_length',
                                                    is_split_into_words=True,
                                                    truncation=True, return_attention_mask=True)
        return tokenized_chunk
    def build_dataset(self, text):
        texts = [[int(s) for s in version.split(' ')] for version in text.split('\n')]
        text_as_int = np.array(list(itertools.chain.from_iterable(texts)))
        x_tensor = self.pytorch_rolling_window(torch.from_numpy(text_as_int[:-1]), self.sequence_length, self.stride)
        y_tensor = self.pytorch_rolling_window(torch.from_numpy(text_as_int[1:]), self.sequence_length, self.stride)
        return x_tensor, y_tensor

    def __len__(self):
        return self.x_input.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.x_input[idx], 'target_ids': self.y_input[idx].type(torch.LongTensor)}

class SymbolicDatasetTruncate(Dataset):
    def __init__(self, symbolic_text, tokenizer, sequence_length=256, stride=100):
        self.tokenizer = tokenizer
        self.stride = stride
        self.sequence_length = sequence_length
        self.x_input, self.y_input = self.build_dataset(symbolic_text, tokenizer, sequence_length, stride)

    def pytorch_rolling_window(self, x, window_size, step_size=1):
        # unfold dimension to make our rolling window
        return x.unfold(0, window_size, step_size)

    def encode_single_piece(self, t):
        piece = t.strip().split()
        piece.append('\n')
        piece.insert(0, '\n')
        max_len = max(len(piece), self.sequence_length)
        tokenized_chunk = self.tokenizer.encode_plus(piece[:max_len], max_length=self.sequence_length, padding='max_length',
                                                    is_split_into_words=True,
                                                    truncation=True, return_attention_mask=True)
        return tokenized_chunk
    def build_dataset(self, text, tokenizer : PreTrainedTokenizerFast, sequence_length, stride):
        list_of_ints = []
        texts = text.split('\n')
        with ThreadPool(64) as p:
            pool_output = p.map(self.encode_single_piece, texts)
        list_of_ints.extend(pool_output)
        del pool_output
        # for t in text.split('\n'):
        #     # piece = t.strip().split()
        #     # piece.append('\n')
        #     # piece.insert(0, '\n')
        #     # chunked_piece = [piece[i:i + sequence_length] for i in range(0, len(piece), sequence_length)]
        #     # for chunk in chunked_piece[:1]:
        #     #     tokenized_chunk = tokenizer.encode_plus(chunk, max_length=sequence_length, padding='max_length', is_split_into_words=True,
        #     #                                   truncation=True, return_attention_mask=True)
        #     #     list_of_ints.append(tokenized_chunk)
        #     list_of_ints.append(self.encode_single_piece(t))

        # list_of_ints = [tokenizer.encode_plus("\n " + t.strip() + " \n", max_length=sequence_length, padding='max_length',
        #                                       truncation=True, return_attention_mask=True) for t in text.split('\n')]
        # text_as_int = np.array([elem for sublist in list_of_ints for elem in sublist])
        return list_of_ints, list_of_ints

    def __len__(self):
        return len(self.x_input)

    def __getitem__(self, idx):
        return {'input_ids': torch.LongTensor(self.x_input[idx].data['input_ids']), 'attention_mask':torch.LongTensor(self.x_input[idx].data['attention_mask']),
                'target_ids':  torch.LongTensor(self.x_input[idx].data['input_ids'])}
class ClassificationDataset(Dataset):
    def __init__(self, datapath, tokenizer: PreTrainedTokenizerFast, seq_len):
        self.xs, self.ys = [], []
        self.tokenizer = tokenizer
        self.build_dataset(datapath, seq_len)
    def build_dataset(self, datapath, seq_len):
        csv_file = open(datapath, "r")
        data = csv.DictReader(csv_file)
        neg_counter = Counter()
        pos_counter = Counter()
        for row in data:
            try:
                label = int(row["label"])
                filepath = row["filepath"]

                data_dir = os.path.dirname(datapath)
                phrase_path = os.path.join(data_dir, filepath) + ".mid"
                encoded_path = os.path.join(data_dir, filepath) + ".npy"

                # Load midi file as text
                text, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)
                ids = [int(s) for s in text.split()]
                if len(ids) > seq_len:
                    print(123)
                    ids = ids[:seq_len]
                    attention_mask = [1 for _ in range(seq_len)]
                else:
                    attention_mask = [1 for _ in range(len(ids))]
                    while len(ids) < seq_len:
                        ids.append(0)
                        attention_mask.append(0)

                # Encode midi text using generative lstm
            #     tokenized_text = self.tokenizer.encode_plus(
            #     '\n ' + text,
            #     max_length=seq_len,
            #     add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            #     return_token_type_ids=False,
            #     padding='max_length',
            #     truncation=True,
            #     return_attention_mask=True,
            #     return_tensors='pt',  # Return PyTorch tensors
            # )
                # Save encoding in file to make it faster to load next time
                # np.save(encoded_path, encoding)
            except Exception as E:
                print(123)
                continue
            if label == -1:
                encoded_label = [1,0]
            else:
                encoded_label = [0,1]
            # tks = self.tokenizer.encode('\n ' + text)
            # for t in tks:
            #     if label == -1:
            #         neg_counter[t] +=1
            #     else:
            #         pos_counter[t] +=1

            self.xs.append({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attention_mask)})
            self.ys.append(torch.tensor(encoded_label))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return {'encoding': self.xs[idx], 'class': self.ys[idx]}


if __name__ == "__main__":
    # Encode midi files as text with vocab
    test_text, test_vocab = me.load('vgmidi/unlabelled/test')
    # Build dictionary to map from char to integers
    tokenizerm = MidiTokenizer(from_file='trained/char2idx.json')
    dataset = SymbolicDataset(test_text, tokenizerm, sequence_length=256, stride=64)
