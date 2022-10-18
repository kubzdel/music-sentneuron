import torch
from torch import nn
from transformers import AutoModel, GPT2Config, GPT2LMHeadModel, BigBirdConfig, BigBirdForCausalLM, \
    GPT2ForSequenceClassification, ReformerConfig, ReformerModelWithLMHead


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
        # config = GPT2Config(
        #     vocab_size=vocab_size,
        #     n_ctx=seq_len,
        #     n_positions=seq_len,
        #     n_layer=n_layer,
        #     n_head=n_head,
        #     n_embd=n_embd,
        #     bos_token_id=110,
        #     eos_token_id=110,
        #     output_hidden_states=True
        # )
        config = BigBirdConfig(
            is_decoder=True,
            vocab_size=vocab_size,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_size=n_embd,
            bos_token_id=110,
            eos_token_id=110,
            output_hidden_states=True,
            max_position_embeddings=1024,
            attention_type='original_full'
        )
        self.gpt2 = AutoModel.from_config(config=config)
        self.linear_layer = nn.Sequential(
            nn.Linear(n_embd, n_embd, bias=True),
            nn.Tanh())
        # # language modelling head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)
        # self.model = GPT2LMHeadModel(config=config)

    def forward(self, tokens):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens.to('cuda')
        gpt2_outputs = self.gpt2(tokens)
        hidden_states = gpt2_outputs[0]
        # equivalent of hidden_states = gpt2_outputs.last_hidden_state
        linear_state = self.linear_layer(hidden_states)
        lm_logits = self.lm_head(linear_state)
        # lm_logits = self.model(tokens).logits
        return lm_logits

class GPT2LM(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd):
        super(GPT2LM, self).__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_ctx=seq_len,
            n_positions=seq_len,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            bos_token_id=104,
            eos_token_id=104,
            pad_token_id=0,
            output_hidden_states=True
        )
        self.gpt2 = GPT2LMHeadModel(config=config)

    def forward(self, tokens, labels=None, attention_mask=None):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens.to('cuda')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda')
        if labels is not None:
            outputs = self.gpt2(tokens,attention_mask=attention_mask, labels=tokens)
            return outputs.logits, outputs.loss
        else:
            outputs = self.gpt2(tokens, attention_mask=attention_mask)
            return outputs.logits


class BigBirdLM(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd):
        super(BigBirdLM, self).__init__()
        # config = GPT2Config(
        #     vocab_size=vocab_size,
        #     n_ctx=seq_len,
        #     n_positions=seq_len,
        #     n_layer=n_layer,
        #     n_head=n_head,
        #     n_embd=n_embd,
        #     bos_token_id=110,
        #     eos_token_id=110,
        #     output_hidden_states=True
        # )
        config = ReformerConfig(
            is_decoder=True,
            vocab_size=vocab_size,
            num_attention_heads=n_head,
            attention_head_size=64,
        hidden_size=n_embd,
            bos_token_id=112,
            axial_pos_shape = [32,64],
        axial_pos_embds_dim= [64, n_embd- 64],
        eos_token_id=112,
            pad_token_id=0,
            output_hidden_states=True,
            max_position_embeddings=2048
        )
        self.bigbird = ReformerModelWithLMHead(config=config)

    def forward(self, tokens, labels=None, attention_mask=None):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens.to('cuda')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda')
        if labels is not None:
            outputs = self.bigbird(tokens,attention_mask=attention_mask, labels=tokens)
            return outputs.logits, outputs.loss
        else:
            outputs = self.bigbird(tokens, attention_mask=attention_mask)
            return outputs.logits


class TransformerClassificationModel(nn.Module):
    def __init__(self, model):
        super(TransformerClassificationModel, self).__init__()
        # self.generative_model = model
        # # language modelling head
        # self.classification_head = nn.Sequential(nn.Linear(768, 768, bias=True),
        # nn.Linear(768, 2, bias=True))
        self.gpt2_class = GPT2ForSequenceClassification.from_pretrained(model, num_labels = 2)
        self.classification_head = nn.Sequential(
            nn.Linear(self.gpt2_class.config.hidden_size, self.gpt2_class.config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.gpt2_class.config.hidden_size * 2, 2)
            # nn.Sigmoid()
        )
        self.gpt2_class.score = self.classification_head
        # for p in self.gpt2_class.transformer.parameters():
        #     p.requires_grad = False
        # self.gpt2_class.classifier = self.classification_head
        # for p in self.gpt2_class.classifier.parameters():
        #     p.requires_grad = True

    def forward(self, input):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        input_ids = torch.squeeze(input['input_ids'], 1)
        attention_mask = torch.squeeze(input['attention_mask'], 1)
        outputs = self.gpt2_class(input_ids, attention_mask=attention_mask)
        # pooled_output = outputs[0]
        # logits = self.classification_head(pooled_output)
        return outputs.logits
        # gpt2_outputs = self.generative_model(input_ids=input_ids, attention_mask=attention_mask)
        # hidden_states = gpt2_outputs.hidden_states[-1]
        # # equivalent of hidden_states = gpt2_outputs.last_hidden_state
        # m = (input_ids == 0).nonzero(as_tuple=True)
        # last_token_embedding = hidden_states[:, -1, :]
        # last_token_embedding = torch.squeeze(last_token_embedding, 1)
        # logits = self.classification_head(last_token_embedding)
        # return logits


class TransformerLstmModel(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd, ckpt=None):
        super(TransformerLstmModel, self).__init__()
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
        # generative_model = ckpt
        # language modelling head
        # self.gpt2 = generative_model.gpt2
        # for p in self.gpt2.parameters():
        #     p.requires_grad = False
        # self.linear_layer = nn.Sequential(
        #     nn.Linear(n_embd, n_embd),
        #     nn.GELU())
        self.lm_head = nn.Linear(512, vocab_size, bias=False)
        self.lstm_module = nn.Sequential(
            nn.LSTM(input_size=n_embd, num_layers=4,
                    hidden_size=512, dropout=0.1, batch_first=True))

    def forward(self, tokens):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        gpt2_outputs = self.gpt2(tokens)
        hidden_states = gpt2_outputs[0]
        lstm_output, (h, c) = self.lstm_module(hidden_states)  ## extract the 1st token's embeddings
        hidden = torch.cat((lstm_output[:, -1, :512], lstm_output[:, 0, 512:]), dim=-1)
        # linear_state = self.linear_layer(hidden_states)
        # # last_token_embedding = hidden_states[:, -1, :]
        # output, (h, c) = self.lstm_module(linear_state)
        # logits = self.lm_head(output)
        # # x_reshaped = logits.permute(0, 2, 1)
        # return logits
        # # equivalent of hidden_states = gpt2_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden)
        return lm_logits
