from torch import nn
from transformers import AutoModel, GPT2Config


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_layers, lstm_units):
        super().__init__()
        self.lstm_module = nn.Sequential(
                nn.Embedding(vocab_size, embedding_size),
                nn.LSTM(input_size=embedding_size, num_layers=lstm_layers,
                        hidden_size=lstm_units, dropout=0.1, batch_first=True))
        self.linear = nn.Linear(lstm_units, vocab_size)

    def forward(self, input):
        output, (h, c) = self.lstm_module(input)
        logits = self.linear(output)
        # x_reshaped = logits.permute(0, 2, 1)
        return logits


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd):
        super(TransformerModel, self).__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_ctx=seq_len,
            bos_token_id=0,
            n_positions=seq_len,
            n_layer=n_layer,
            n_head=n_head,
            eos_token_id=0,
            n_embd=n_embd,
            output_hidden_states=True
        )
        self.gpt2 = AutoModel.from_config(config=config)
        # language modelling head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, tokens):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        gpt2_outputs = self.gpt2(tokens)
        hidden_states = gpt2_outputs[0]
        # equivalent of hidden_states = gpt2_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

