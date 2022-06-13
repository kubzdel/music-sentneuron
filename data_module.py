import pytorch_lightning as pl
from torch.utils.data import DataLoader

from symbolic_dataset import SymbolicDataset


class MusicDataModule(pl.LightningDataModule):
    def __init__(self, train_input, val_input, tokenizer, sequence_length, stride, batch_size=32, num_workers=0):
        super().__init__()
        self.train_input = train_input
        self.val_input = val_input
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride


    def _build_default_custom_dataset(self, symbolic_text):
        return SymbolicDataset(symbolic_text, self.tokenizer, self.sequence_length, self.stride)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self._build_default_custom_dataset(self.train_input)
            self.valid_dataset = self._build_default_custom_dataset(self.val_input)
        if stage == 'test' or stage is None:
            self.test_dataset = self._build_default_custom_dataset(self.val_input)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)
