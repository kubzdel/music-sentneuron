import json
import os
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace, Split
from tokenizers.trainers import BpeTrainer, WordLevelTrainer


class MidiTokenizer:
    def __init__(self, vocab=None, from_file = None):
        if from_file:
            with open(from_file, 'r') as f:
                self.mapping = json.load(f)
        else:
            self.mapping = {char: i for i, char in enumerate(vocab)}
        self.demapping =  {idx:char for char,idx in self.mapping.items()}
    def encode(self, text, return_tensors = False):
        if return_tensors:
            return torch.from_numpy(np.array([self.mapping[c] for c in text.split(" ")]))
        return np.array([self.mapping[c] for c in text.split(" ")])
    def decode(self, ids):
        ids_lst = ids.tolist()
        decoded = ' '.join([self.demapping[c] for c in ids_lst])
        return decoded



def count_tokens():
    counter = Counter()
    paths = [str(x) for x in Path(".").glob("clean/unlabelled/train/*.txt")]
    paths+=([str(x) for x in Path(".").glob("clean/unlabelled/test/*.txt")])
    unfiltered = 0
    filtered= 0
    for p in paths:
        with open(p) as f:
            lines = f.readlines()
            for line in lines:
                unfiltered += 1
                for word in line.split():
                    counter[word] +=1
    print(counter.most_common())
    min_occ = {x: count for x, count in counter.items() if count >= 2000 or 'w_' not in x}
    ws = [counter.most_common()]
    print(len(counter))
    print(len(min_occ))
    for p in paths:
        with open(p) as f:
            lines = f.readlines()
            for line in lines:
                if any(w not in min_occ for w in line.split()):
                    continue
                else:
                    filtered +=1

    print(unfiltered)
    print(filtered)
    print(str(filtered/unfiltered))
    print([x for x, count in counter.items() if count < 2000 and 'w_' in x])

def train_tokenizer(from_file=None):
    if from_file:
        with open(from_file, 'r') as f:
            mapping = json.load(f)
    # paths = []
    paths = [str(x) for x in Path(".").glob("clean/unlabelled/train/*.txt")]
    paths+=([str(x) for x in Path(".").glob("clean/unlabelled/test/*.txt")])

    # Initialize a tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=mapping, unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Split(pattern=" ",behavior="removed")
    # Customize training
    trainer = WordLevelTrainer(special_tokens=["<PAD>", "<UNK>"])
    # tokenizer.add_special_tokens(["<PAD>","<UNK>"])
    tokenizer.enable_padding(pad_token="<PAD>")
    tokenizer.train(paths, trainer)
    tokenizer.save("new_tokenizer_word_big.json")
    print(tokenizer.encode("n_61 n_61 n_61  abcd \n").ids)
    return tokenizer

if __name__ == "__main__":
    train_tokenizer(from_file='trained/char2idx_big.json')
    # count_tokens()