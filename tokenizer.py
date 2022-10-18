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

import midi_encoder


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
    paths = [str(x) for x in Path(".").glob("vgmidi/unlabelled/train/*.txt")]
    paths+=([str(x) for x in Path(".").glob("vgmidi/unlabelled/test/*.txt")])
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
    min_occ = {x: count for x, count in counter.items() if count >= 200 or 'w_' not in x}
    blck = [x for x, count in counter.items() if 'w_' in x and int(x.split('_')[1]) > 128]
    blck2 = [x for x, count in counter.items() if count < 150 and 'w_' in x]
    print(len(blck))
    print(len(blck2))
    ws = [counter.most_common()]
    print(len(counter))
    print(len(min_occ))
    for p in paths:
        with open(p) as f:
            lines = f.readlines()
            for line in lines:
                if any(w  in ['w_152'] for w in line.split()):
                    print(p)
                    continue
                else:
                    filtered +=1

    print(unfiltered)
    print(filtered)
    print(str(filtered/unfiltered))
    blck = [x for x, count in counter.items() if ((count < 200 and 'w_' in x) or ('w_' in x and int(x.split('_')[1]) > 128))]
    print(len(blck))
    print(blck)

    # print([x for x, count in counter.items() if 'w_' in x and int(x.split('_')[1]) > 128])

def train_tokenizer(from_file=None):
    if from_file:
        with open(from_file, 'r') as f:
            mapping = json.load(f)
    # paths = []
    paths = [str(x) for x in Path(".").glob("vgmidi/unlabelled/train/*.txt")]
    paths+=([str(x) for x in Path(".").glob("vgmidi/unlabelled/test/*.txt")])

    # Initialize a tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=mapping, unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Split(pattern=" ",behavior="removed")
    # Customize training
    trainer = WordLevelTrainer(special_tokens=["<PAD>", "<UNK>"])
    # tokenizer.add_special_tokens(["<PAD>","<UNK>"])
    tokenizer.enable_padding(pad_token="<PAD>")
    tokenizer.enable_truncation(max_length=2048)
    tokenizer.train(paths, trainer)
    tokenizer.save("new_tokenizer_word_old.json")
    print(tokenizer.encode("n_61 n_61 n_61  abcd \n").ids)
    return tokenizer

def generate_midi_from_txt():
        # paths = []
    paths = [str(x) for x in Path(".").glob("vgmidi/unlabelled/test/*.txt")]
    paths+=([str(x) for x in Path(".").glob("vgmidi/unlabelled/train/*.txt")])
    unfiltered = 0
    filtered= 0
    for p in paths:
        if "Final_Fantasy_7_LurkingInTheDarkness" in p:
            with open(p) as f:
                line = f.readlines()[0].strip()
                new_name = p.split('.txt')[0]+'_decoded.mid'
                midi_encoder.write(line,new_name)
from miditok import REMI
def generate_remi_tokens(path):
    additional_tokens = {'Chord': True, 'Rest': False, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,  # nb of tempo bins
                         'tempo_range': (40, 250)}  # (min, max)
    tokenizer = REMI(additional_tokens=additional_tokens, mask=False)
    def midi_valid(midi) -> bool:
        if any(ts.numerator != 4 for ts in midi.time_signature_changes):
            return False  # time signature different from 4/*, 4 beats per bar
        if midi.max_tick < 10 * midi.ticks_per_beat:
            return False  # this MIDI is too short
        return True
    paths = [str(x) for x in Path(".").glob(f"{path}/*.mid")]

    # Converts MIDI files to tokens saved as JSON files
    tokenizer.tokenize_midi_dataset(paths, path, midi_valid)
if __name__ == "__main__":
    # generate_midi_from_txt()
    # train_tokenizer(from_file='trained/char2idx_sturm.json')
    generate_remi_tokens("vgmidi/unlabelled/test")
    # count_tokens()