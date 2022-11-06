import math
from abc import ABC
import torch

class SentimentController(ABC):
    def __init__(self, mode):
        self.mode = mode
        self.controlled_token_ids = {}
    def apply(self, seq):
        controlled_ids = list(self.controlled_token_ids[self.mode])
        seq = torch.squeeze(seq, 0)
        seq[(torch.tensor(controlled_ids),)] = -math.inf
        seq = torch.unsqueeze(seq,0)
        return seq

class TempoController(SentimentController):
    def __init__(self, mode):
        super().__init__(mode)
        self.controlled_token_ids = {"slow": range(256, 279), "fast": range(247, 265)}

class PitchController(SentimentController):
    def __init__(self, mode):
        super().__init__(mode)
        self.controlled_token_ids = {"low": range(44, 92), "high": range(4, 34)}
