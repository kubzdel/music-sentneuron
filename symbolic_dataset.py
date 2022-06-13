# from tensorflow_core.contrib.data.python.ops import sliding
from torch.utils.data.dataset import Dataset
import torch
import midi_encoder as me
from tokenizer import MidiTokenizer


class SymbolicDataset(Dataset):
    def __init__(self, symbolic_text, tokenizer, sequence_length = 256, stride = 100):
        self.tokenizer = tokenizer
        self.stride = stride
        self.sequence_length = sequence_length
        self.x_input, self.y_input = self.build_dataset(symbolic_text, tokenizer, sequence_length, stride)

    def pytorch_rolling_window(self, x, window_size, step_size=1):
        # unfold dimension to make our rolling window
        return x.unfold(0, window_size, step_size)
    def build_dataset(self, text, tokenizer, sequence_length, stride):
            text_as_int = tokenizer.encode(text)
            x_tensor = self.pytorch_rolling_window(torch.from_numpy(text_as_int[:-1]), sequence_length, stride)
            y_tensor = self.pytorch_rolling_window(torch.from_numpy(text_as_int[1:]), sequence_length, stride)
            return x_tensor, y_tensor
    def __len__(self):
        return self.x_input.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.x_input[idx], 'target_ids':self.y_input[idx].type(torch.LongTensor)}

if __name__ == "__main__":
    # Encode midi files as text with vocab
        test_text, test_vocab = me.load('vgmidi/unlabelled/test')
        # Build dictionary to map from char to integers
        tokenizer = MidiTokenizer(from_file='trained/char2idx.json')
        dataset = SymbolicDataset(test_text, tokenizer,sequence_length=256, stride = 64)