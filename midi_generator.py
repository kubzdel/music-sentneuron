import os
import json
import argparse
import numpy as np
# import midi_encoder as me
# from train_generative import build_generative_model
# from train_classifier import preprocess_sentence

from tokenizer import MidiTokenizer
from transformer_generative import MidiTrainingModule
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer
from dotenv import load_dotenv

GENERATED_DIR = './generated'


# def override_neurons(model, layer_idx, override):
#     h_state, c_state = model.get_layer(index=layer_idx).states
#
#     c_state = c_state.numpy()
#     for neuron, value in override.items():
#         c_state[:,int(neuron)] = int(value)
#
#     model.get_layer(index=layer_idx).states = (h_state, tf.Variable(c_state))
#
#
# def sample_next(predictions, k):
#     # Sample using a categorical distribution over the top k midi chars
#     top_k = tf.math.top_k(predictions, k)
#     top_k_choices = top_k[1].numpy().squeeze()
#     top_k_values = top_k[0].numpy().squeeze()
#
#     if np.random.uniform(0, 1) < .5:
#         predicted_id = top_k_choices[0]
#     else:
#         p_choices = tf.math.softmax(top_k_values[1:]).numpy()
#         predicted_id = np.random.choice(top_k_choices[1:], 1, p=p_choices)[0]
#
#     return predicted_id
#
#
# def process_init_text(model, init_text, char2idx, layer_idx, override):
#     model.reset_states()
#
#     for c in init_text.split(" "):
#         # Run a forward pass
#         try:
#             input_eval = tf.expand_dims([char2idx[c]], 0)
#
#             # override sentiment neurons
#             override_neurons(model, layer_idx, override)
#
#             predictions = model(input_eval)
#         except KeyError:
#             if c != "":
#                 print("Can't process char", s)
#
#     return predictions
#
#
# def generate_midi(model, char2idx, idx2char, init_text="", seq_len=256, k=3, layer_idx=-2, override={}):
#     # Add front and end pad to the initial text
#     init_text = preprocess_sentence(init_text)
#
#     # Empty midi to store our results
#     midi_generated = []
#
#     # Process initial text
#     predictions = process_init_text(model, init_text, char2idx, layer_idx, override)
#
#     # Here batch size == 1
#     model.reset_states()
#     for i in range(seq_len):
#         # remove the batch dimension
#         predictions = tf.squeeze(predictions, 0).numpy()
#
#         # Sample using a categorical distribution over the top k midi chars
#         predicted_id = sample_next(predictions, k)
#
#          # Append it to generated midi
#         midi_generated.append(idx2char[predicted_id])
#
#         # override sentiment neurons
#         override_neurons(model, layer_idx, override)
#
#         #Run a new forward pass
#         input_eval = tf.expand_dims([predicted_id], 0)
#         predictions = model(input_eval)
#
#     return init_text + " " + " ".join(midi_generated)

def get_latest_checkpoint_file(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def grid_search(top_ks, top_ps, temperatures, typical_tau):
    for top_k in top_ks:
        for top_p in top_ps:
            for temperature in temperatures:
                for tau in typical_tau:
                    yield top_k, top_p, temperature, tau


def sampling_testing_grid_search(top_ks, top_ps, temperatures, typical_taus):
    """
    Generate midi file for each combination of parameters of the model.
    :param top_ks: list of top_k values to test
    :param top_ps: list of top_p values to test
    :param temperatures: list of temperatures to test
    :param typical_taus: list of typical taus to test
    """
    tokenizer = MidiTokenizer(from_file='trained/char2idx.json')
    checkpoint = get_latest_checkpoint_file(opt.model)
    load_dotenv()

    for top_k, top_p, temperature, tau in grid_search(top_ks, top_ps, temperatures, typical_taus):
        model = MidiTrainingModule.load_from_checkpoint(checkpoint, tokenizer=tokenizer,
                                                        top_k=top_k, top_p=top_p, temperature=temperature,
                                                        typical_tau=tau)
        model.seq_len = opt.seqlen
        mlf_logger = MLFlowLogger(experiment_name="sampling-midi", tracking_uri="http://150.254.131.193:5000/",
                                  run_name=opt.run_name)
        trainer = Trainer(logger=mlf_logger)
        trainer.predict(model=model)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_generator.py')
    parser.add_argument('--run_name', type=str, required=True, help='MLFlow run name')
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--model_type', type=str, default='transformer', help="Transformer or LSTM.")
    parser.add_argument('--n_head', type=int, default=512, help="Transformer heads.")
    parser.add_argument('--units', type=int, default=512, help="LSTM units.")
    parser.add_argument('--layers', type=int, default=4, help="LSTM layers.")
    parser.add_argument('--seqinit', type=str, default="\n", help="Sequence init.")
    parser.add_argument('--seqlen', type=int, default=256, help="Sequence length.")
    parser.add_argument('--cellix', type=int, default=-2, help="LSTM layer to use as encoder.")
    parser.add_argument('--override', type=str, default="", help="JSON file with neuron values to override.")
    parser.add_argument('--grid_search', type=bool, default=False, help="Run a grid search.")
    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Load override dict from json file
    override = {}

    try:
        with open(opt.override) as f:
            override = json.load(f)
    except FileNotFoundError:
        print("Override JSON file not provided.")

    # Create idx2char from char2idx dict
    idx2char = {idx: char for char, idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild model from checkpoint
    if opt.model_type.lower() == 'transformer':
        if opt.grid_search:
            sampling_testing_grid_search(top_ks=[0], top_ps=[1.0], temperatures=[1.4, 1.5],
                                         typical_taus=[0.2, 0.25, 0.3, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        else:
            tokenizer = MidiTokenizer(from_file='trained/char2idx.json')
            checkpoint = get_latest_checkpoint_file(opt.model)
            model = MidiTrainingModule.load_from_checkpoint(checkpoint, tokenizer=tokenizer,
                                                            top_k=0, top_p=1, temperature=1.5, typical_tau=0.7)
            model.seq_len = opt.seqlen
            mlf_logger = MLFlowLogger(experiment_name="sampling-midi", tracking_uri="http://150.254.131.193:5000/",
                                      run_name=opt.run_name)
            load_dotenv()
            trainer = Trainer(logger=mlf_logger)
            midi_txt = trainer.predict(model=model, dataloaders=None)
            print(midi_txt)

    elif opt.model_type.lower() == 'lstm':
        # TODO - implement pl LSTM support
        # model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
        # model.load_weights(tf.train.latest_checkpoint(opt.model))
        # model.build(tf.TensorShape([1, None]))
        midi_txt = 'LSTM not implemented yet'

    else:
        print("Invalid model type.")
        exit(1)

    # Generate a midi as text
    # midi_txt = generate_midi(model, char2idx, idx2char, opt.seqinit, opt.seqlen, layer_idx=opt.cellix, override=override)
    # print(midi_txt)

    # me.write(midi_txt, os.path.join(GENERATED_DIR, "generated.mid"))
