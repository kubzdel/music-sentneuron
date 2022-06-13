import json
import torch
import numpy as np



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