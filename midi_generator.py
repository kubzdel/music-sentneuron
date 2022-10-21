import argparse
import json
import os

import numpy as np
import torch.nn.functional as F
from miditok import REMI
from transformers import PreTrainedTokenizerFast, TypicalLogitsWarper, RepetitionPenaltyLogitsProcessor, \
    TopKLogitsWarper, TopPLogitsWarper

from sentiment_controllers import TempoController
from transformer_classifier import MidiClassificationModule
from transformer_generative import MidiTrainingModule
import midi_encoder as me
# tf.config.set_visible_devices([], 'GPU')
import torch

from train_classifier import preprocess_sentence

GENERATED_DIR = './generated'


def override_neurons(model, layer_idx, override):
    # pass
    # h_state, c_state = model.get_layer(index=layer_idx).states
    hidden_states = model.linear_layer[0]

    hidden_states_weights = hidden_states.weight
    for neuron, value in override.items():
        with torch.no_grad():
            # @TODO verify value
            hidden_states_weights[:, int(neuron)] = value
    # with torch.no_grad():
    #     model.linear_l
    # model.get_layer(index=layer_idx).states = (h_state, tf.Variable(c_state))


def sample_next(predictions, k):
    # Sample using a categorical distribution over the top k midi chars
    typical_logits_wraper = TypicalLogitsWarper(mass=0.8)
    lm_logits = torch.from_numpy(predictions / 1.2)
    filtered_logits = typical_logits_wraper(torch.LongTensor(), lm_logits)[-1, :]
    probabilities = F.softmax(filtered_logits, dim=-1)
    predicted_token = torch.multinomial(probabilities, 1)
    return predicted_token.cpu().item()


def process_init_text(model, init_text, tokenizer, layer_idx, override):
    # model.reset_states()

    for c in init_text.split(" "):
        # Run a forward pass
        try:
            # input_eval = torch.([char2idx[c]], 0)
            input_eval = torch.unsqueeze(torch.tensor([tokenizer[c]]), 0)

            # override sentiment neurons
            # override_neurons(model, layer_idx, override)
            predictions = model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", s)

    return predictions


def generate_midi(model, tokenizer, idx2char, init_text="", seq_len=256, k=5, layer_idx=-2, override={}):
    # Add front and end pad to the initial text
    init_text = preprocess_sentence(init_text)

    # Empty midi to store our results
    midi_generated = []

    # Process initial text
    predictions = process_init_text(model, init_text, tokenizer, layer_idx, override)

    # Here batch size == 1
    # model.reset_states()
    predicted_ids = []
    for i in range(seq_len):
        with torch.no_grad():
            # remove the batch dimension
            predictions = torch.squeeze(predictions, 0).cpu().detach().numpy()
            if predictions.shape[-2] > 1:
                predictions = predictions[-1, :]
                predictions = np.expand_dims(predictions, 0)
            # Sample using a categorical distribution over the top k midi chars
            predicted_id = sample_next(predictions, k)
            predicted_ids.append(predicted_id)
            # Append it to generated midi
            midi_generated.append(idx2char[predicted_id])

            # override sentiment neurons
            override_neurons(model, layer_idx, override)

            # Run a new forward pass
            # input_eval = tf.expand_dims([predicted_id], 0)
            input_eval = torch.unsqueeze(torch.tensor(predicted_ids), 0)
            predictions = model(input_eval)
    # predicted_token = torch.Tensor([1])
    # with torch.no_grad():
    #     output_seq = torch.as_tensor(tokenizer.encode('\n'))
    #     while (
    #             len(output_seq) < 512 - 1 and predicted_token.unsqueeze(-1)[0] != 99
    #     ):
    #         outputs = model.forward(output_seq).cpu().detach()
    #         lm_logits = outputs
    #         logits = lm_logits[-1, :]
    #         top_k, top_p, temperature = 0, 0.95, 1.4
    #         filtered_logits = top_k_top_p_filtering(
    #             logits, top_k=top_k, top_p=top_p, temperature=temperature
    #         )
    #         probabilities = F.softmax(filtered_logits, dim=-1)
    #         probabilities_logits, probabilities_position = torch.sort(
    #             probabilities, descending=True
    #         )
    #         predicted_token = torch.multinomial(probabilities, 1)
    #         predicted_token = tokenizer.decode(predicted_token)
    #         # predicted_token = re.sub(r'w_\w+\b', 'w_2', predicted_token)
    #         predicted_token = torch.as_tensor(tokenizer.encode(predicted_token))
    #         output_seq = torch.cat([output_seq, predicted_token])
    #     output_seq = output_seq[1:-1]
    #     output_sentence = tokenizer.decode(output_seq)
    #     # output_sentence = re.sub(r'w_\w+\b', 'w_2', output_sentence)
    #     return output_sentence

    return init_text + " " + " ".join(midi_generated)

def generate_midi_with_sentiment(model, classifier, sentiment, tokenizer, idx2char, init_text="", seq_len=256, k=3, sent_controllers=[], beam_size=4):
    # Add front and end pad to the initial text
    # init_text = preprocess_sentence(init_text)

    # Empty midi to store our results
    midi_generated = []

    # Process initial text
    layer_idx = 5
    # predictions = process_init_text(model, init_text, tokenizer, layer_idx, override)
    chunk_size = 250
    # Here batch size == 1
    # model.reset_states()
    beams = [[] for i in range(beam_size)]
    wrap1 = TopPLogitsWarper(top_p=0.95)
    wrap2 = TopKLogitsWarper(top_k=30)
    typical_logits_wraper = TypicalLogitsWarper(mass=0.7)
    logits_penalty = RepetitionPenaltyLogitsProcessor(penalty=1.2)
    # beams = np.empty((beam_size, 1), dtype=np.int32)
    # start_seq = tokenizer.encode("\n", return_tensors='pt').to('cuda')
    start_seq = [2]
    # model.to('cuda')
    while(len(beams[0])) < 1024:
        beams_prob = [1 for i in range(beam_size)]
        for beam in range(beam_size):
            beams[beam] = start_seq.copy()
            for i in range(chunk_size):
                with torch.no_grad():
                    # torch.manual_seed(beam)
                    # Run a new forward pass
                    # input_eval = tf.expand_dims([predicted_id], 0)
                    input_eval = torch.tensor(beams[beam])
                    # if input_eval.ndim < 2:
                    input_eval = torch.unsqueeze(input_eval, 0)
                    predictions = model(input_eval)
                    # remove the batch dimension
                    predictions = torch.squeeze(predictions, 0).cpu().detach().numpy()
                    if predictions.shape[0] > 1:
                        predictions = predictions[-1, :]
                        predictions = np.expand_dims(predictions, 0)
                    # Sample using a categorical distribution over the top k midi chars
                    # predicted_id = sample_next(predictions, k)
                    logits_given_prev = torch.from_numpy(predictions / 1.3)
                    for s_controller in sent_controllers:
                        logits_given_prev = s_controller.apply(logits_given_prev)
                    # filtered_logits = wrap1(input_eval, logits_given_prev)
                    # filtered_logits = wrap2(input_eval, filtered_logits)
                    filtered_logits = typical_logits_wraper(input_eval, logits_given_prev)
                    filtered_logits = logits_penalty(input_eval, filtered_logits)
                    log_prob_char_given_prev = F.softmax(filtered_logits[-1, :], dim=-1).detach().cpu()
                    predicted_id = torch.multinomial(log_prob_char_given_prev, 1).item()
                    predicted_prob = log_prob_char_given_prev[predicted_id].item()
                    beams_prob[beam] *= predicted_prob
                    # s_leng = input_eval.size(1)
                    # beam_output = model.gpt2.generate(
                    #     input_eval.to('cuda'),
                    #     max_length=512,
                    #     temperature=1.2,
                    #     num_return_sequences=1,
                    #     do_sample=True,
                    #     top_k=40,
                    #     top_p=0.99,
                    #     early_stopping=False
                    # )
                    # predicted_id = beam_output[0].cpu().detach().tolist()
                    beams[beam].append(predicted_id)
                    # beams[beam] = predicted_id
                    # Append it to generated midi
                    # for id in predicted_id:
                    #     midi_generated.append(idx2char[id])
                    midi_generated.append(idx2char[predicted_id])
                    if len(beams[beam]) >= 1024:
                        break
        stacked_classifications = []
        # beam_output = model.gpt2.generate(
        #     start_seq,
        #     max_length=90,
        #     num_beams=5,
        #     do_sample=True,
        #     temperature=1.8,
        #     num_return_sequences=5,
        #     early_stopping=True
        # ).cpu().detach().numpy()
        for beam in range(beam_size):
            # beam_txt = ' '.join(idx2char[w] for w in beams[beam][-chunk_size:])
            # ids = [int(s) for s in beam_txt.split()]
            ids = beams[beam][-chunk_size:]
            if ids[-1] != 3:
                ids.append(3)
            if len(ids) > seq_len:
                print(123)
                ids = ids[:seq_len]
                attention_mask = [1 for _ in range(seq_len)]
            else:
                attention_mask = [1 for _ in range(len(ids))]
                while len(ids) < seq_len:
                    ids.append(0)
                    attention_mask.append(0)
            classification_input = {"input_ids": torch.unsqueeze(torch.tensor(ids),0), "attention_mask": torch.unsqueeze(torch.tensor(attention_mask),0)}
            classification_output = F.softmax(classification_model(classification_input)).detach().cpu().numpy()[0]
            stacked_classifications.append(classification_output)
        stacked_classifications = np.array(stacked_classifications)
        # stacked_classifications_ids = np.where(stacked_classifications[:, sentiment] > 0.80)[0]
        # filtered_proba = [beams_prob[i] if i in stacked_classifications_ids else 0 for i in range(beam_size)]
        # max_item = max(filtered_proba)
        k = np.argmax(stacked_classifications[:, sentiment], axis=0)
        # k = filtered_proba.index(max_item)
        start_seq = beams[k]
    # predicted_token = torch.Tensor([1])
    # with torch.no_grad():
    #     output_seq = torch.as_tensor(tokenizer.encode('\n'))
    #     while (
    #             len(output_seq) < 512 - 1 and predicted_token.unsqueeze(-1)[0] != 99
    #     ):
    #         outputs = model.forward(output_seq).cpu().detach()
    #         lm_logits = outputs
    #         logits = lm_logits[-1, :]
    #         top_k, top_p, temperature = 0, 0.95, 1.4
    #         filtered_logits = top_k_top_p_filtering(
    #             logits, top_k=top_k, top_p=top_p, temperature=temperature
    #         )
    #         probabilities = F.softmax(filtered_logits, dim=-1)
    #         probabilities_logits, probabilities_position = torch.sort(
    #             probabilities, descending=True
    #         )
    #         predicted_token = torch.multinomial(probabilities, 1)
    #         predicted_token = tokenizer.decode(predicted_token)
    #         # predicted_token = re.sub(r'w_\w+\b', 'w_2', predicted_token)
    #         predicted_token = torch.as_tensor(tokenizer.encode(predicted_token))
    #         output_seq = torch.cat([output_seq, predicted_token])
    #     output_seq = output_seq[1:-1]
    #     output_sentence = tokenizer.decode(output_seq)
    #     # output_sentence = re.sub(r'w_\w+\b', 'w_2', output_sentence)
    #     return output_sentence
    # out =' '.join(idx2char[w] for w in start_seq)
    return start_seq
def get_top_n(arr, n):
    flat = arr.flatten()
    idxs = np.argpartition(flat, -n)[-n:]
    idxs = idxs[np.argsort(-flat[idxs])]
    ps = flat[idxs]
    idxs = np.unravel_index(idxs, arr.shape)
    return idxs, ps

def generate_sentiment(model, classifier, sentiment, beam_width, chard2idx, idx2char):

    beam_seq = np.empty((beam_width, 1), dtype=np.int32)

    beam_log_probs = np.zeros((beam_width,1))

    vocab_length = len(chard2idx)
    prob_char_given_prev = np.empty((beam_width, vocab_length))
    typical_logits_wraper = TypicalLogitsWarper(mass=0.8)
    done = False
    first_char = True
    while not done:

        if first_char:
            input_eval = torch.unsqueeze(torch.tensor([tokenizer.vocab['\n']]), 0)
            prob_first_char = model(input_eval)
            prob_first_char = torch.squeeze(prob_first_char, 0)
            if prob_first_char.shape[-2] > 1:
                prob_first_char = prob_first_char[-1, :]
                prob_first_char = np.expand_dims(prob_first_char, 0)
            log_prob_first_char = torch.squeeze(F.softmax(prob_first_char, dim=-1),0).cpu().detach().numpy()
            top_n, log_p = get_top_n(log_prob_first_char, beam_width)
            beam_seq[:,0] = top_n[0]
            beam_log_probs[:,0] += log_p
            first_char = False
        else:

            for beam in range(beam_width):
                input_eval = torch.from_numpy(beam_seq[beam]).to('cuda')
                model_output = model(input_eval).detach().cpu().numpy()[-1]
                prob_char_given_prev[beam] =  model_output
            logits_given_prev = torch.from_numpy(prob_char_given_prev / 1.1)
            filtered_logits = typical_logits_wraper(torch.LongTensor(), logits_given_prev)
            log_prob_char_given_prev =  F.softmax(filtered_logits, dim=-1).detach().cpu().numpy()
            # predicted_token = torch.multinomial(log_prob_char_given_prev, 1)
            log_prob_char = beam_log_probs + log_prob_char_given_prev
            top_n, log_p = get_top_n(log_prob_char, beam_width)
            beam_seq = np.hstack((beam_seq[top_n[0]], top_n[1].reshape(-1,1)))

            beam_log_probs = log_p.reshape(-1,1)

        if len(beam_seq[0]) == 512:
            done = True

    return beam_seq

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--seqinit', type=str, default="\n", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence lenght.")
    parser.add_argument('--cellix', type=int, default=-2, help="LSTM layer to use as encoder.")
    parser.add_argument('--override', type=str, default="", help="JSON file with neuron values to override.")
    opt = parser.parse_args()
    # Load override dict from json file
    override = {}

    try:
        with open(opt.override) as f:
            override = json.load(f)
    except FileNotFoundError:
        print("Override JSON file not provided.")

    # tokenizer = PreTrainedTokenizerFast(tokenizer_file="new_tokenizer_word_sturm.json")
    # tokenizer.unk_token = "<UNK>"
    # tokenizer.pad_token = "<PAD>"
    # vocab_size = len(tokenizer)
    pitch_range = range(21, 110)
    beat_res ={(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False, 'Bar':False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,
                         'tempo_range': (30, 200)}  # (min, max)
    sos_eos_tokens = ['SOS']
    remi_tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens = True, mask=False)
    vocab_size = len(remi_tokenizer)
    char2idx = remi_tokenizer.vocab.event_to_token
    # Create idx2char from char2idx dict
    idx2char = remi_tokenizer.vocab.token_to_event

    # Calculate vocab_size from char2idx dict
    # vocab_size = len(char2idx)
    sentiment = 1
    # Rebuild generative model from checkpoint
    # gen_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    # gen_model.load_weights(tf.train.latest_checkpoint(opt.genmodel))
    # gen_model.build(tf.TensorShape([1, None]))
    model = MidiTrainingModule.load_from_checkpoint('gpt2_remi_new_data/last.ckpt', batch_size=4,
                                                               epochs=2, samples_count = 100, tokenizer = None, embedding_size = 10,
                                                               vocab_size = vocab_size, lstm_layers = 3, lstm_units = 3,
                                                               n_layer=6, n_head = 6, n_embd = 402, seq_len=1024, training_stride=1024, validation_stride=1024).model
    model.eval()
    model.to('cuda')
    # beam_search_output = model.gpt2.generate(tokenizer.encode('\n', return_tensors='pt').to('cuda'),
    #                                     max_length=100,
    #                                     num_beams=5,
    #                                     do_sample=False,
    #                                     no_repeat_ngram_size=2)
    classification_model = MidiClassificationModule.load_from_checkpoint('test_clssifier/epoch=5-step=161-v1.ckpt')
    classification_model.eval()
    # classification_model.to('cuda')
    # midi_token_arrays = generate_sentiment(model=model, classifier=classification_model, sentiment=sentiment, beam_width=5,
    #                               chard2idx=char2idx, idx2char=idx2char)
    # gen_token_ids = midi_token_arrays[0]
    # gen_events = tokenizer.decode(gen_token_ids)
    # print(123)



    # Rebuild model from checkpoint
    # model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    # model.load_weights(tf.train.latest_checkpoint(opt.model))
    # model.build(tf.TensorShape([1, None]))

    # Generate a midi as text
    # midi_txt = generate_midi(model, char2idx, idx2char, opt.seqinit, opt.seqlen, layer_idx=opt.cellix, override=override)

    # print(midi_txt)
    #

    sent_controllers = [TempoController(mode="fast")]
    for i in range(5):
        midi_txt = generate_midi_with_sentiment(model, classification_model, sentiment, char2idx, idx2char, opt.seqinit,
                                                opt.seqlen, sent_controllers=sent_controllers, beam_size=10)
        print(midi_txt)
        output_sentence_midi = remi_tokenizer.tokens_to_midi([midi_txt])
        # Path(f'{self.run_name}_generated_samples').mkdir(exist_ok=True)
        # midi_path = f'{self.run_name}_generated_samples/{self.run_name}_step_{str(step)}.mid'
        # write(output_sentence, midi_path)
        output_sentence_midi.dump(os.path.join(GENERATED_DIR, f"generated_{i}.mid"))
        # me.write(midi_txt, os.path.join(GENERATED_DIR, f"generated_{i}.mid"))
