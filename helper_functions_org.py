import os
from pathlib import PurePosixPath
import torch
from minio import Minio
from cachetools.func import lru_cache

import onnxruntime
import torch.nn.functional as F
from utils import top_k_top_p_filtering
from miditok import REMI
from transformers import PreTrainedTokenizerFast, TypicalLogitsWarper, RepetitionPenaltyLogitsProcessor, \
    TopKLogitsWarper, TopPLogitsWarper
import numpy as np

SEQ_LEN = 1024


@lru_cache
def load_onnx_model(model_path):
  return onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])


def to_numpy(tensor):
  return tensor.detach().numpy()

@lru_cache
def download_metadata(model_to_download):
  #MODEL_NAME = os.getenv('MODEL_NAME')  # -środowiskowa
  # odwortne do export model
  #download model and tokenizer then load locally
  client = Minio("150.254.131.193:9000", access_key=os.environ["AWS_ACCESS_KEY_ID"],
                 secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], secure=False)
  model_bucket_path = str(PurePosixPath('deployment', '{}'.format(model_to_download)))  # aws_path

  if not os.path.exists("models"):
    os.makedirs("models")

  target_model_name = os.path.join("models", model_to_download)

  if not os.path.isfile(target_model_name):
    client.fget_object(os.environ["AWS_BUCKET_NAME"], model_bucket_path, f"{target_model_name}")
  # tokenizer
  # token_bucket_path = str(PurePosixPath('deployment', '{}'.format("char2idx.json")))
  # tokenizer_file = client.fget_object(os.environ["AWS_BUCKET_NAME"], token_bucket_path, "downloaded_tokenizer.json")

  return f"{target_model_name}"#, tokenizer_file, "downloaded_tokenizer.json"

def generate_midi_file(model_to_download):
  #wczytanie modelu z bucketa (potrzebny model i tokenizer (dane na buckecie)
  model_path = download_metadata(model_to_download)    #, tokenizer, tokenizer_path

  pitch_range = range(21, 110)
  beat_res = {(0, 4): 8, (4, 12): 4}
  nb_velocities = 32
  additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                       'Bar': False,
                       'rest_range': (2, 8),  # (half, 8 beats)
                       'nb_tempos': 32,
                       'tempo_range': (30, 200)}  # (min, max)
  remi_tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens=True, mask=False)

  # wczytanie modelu
  ort_session = load_onnx_model(model_path)
  midi_path = inference_model(ort_session, remi_tokenizer)


  return midi_path

def inference_model(onnxsession, remi_tokenizer, start_seq=[2]):  #inference_model(ort_session, tokenizer, model_path)

  output_seq = torch.tensor(start_seq)
  #output_seq = Tokenizer.generate_midi_from_txt(start_seq)#.encode(start_seq, return_tensors=True)
  predicted_token = torch.Tensor([1])
  while (
          len(output_seq) < SEQ_LEN - 1 and predicted_token.unsqueeze(-1)[0] != 0
  ):
    onnx_input = {"input_tokens": to_numpy(torch.unsqueeze(output_seq, 0))}

    outputs = onnxsession.run(None, onnx_input)[0]
    #outputs = self.forward(torch.unsqueeze(output_seq, 0), permute=False)

    # outputs = self.forward(output_seq, permute=False)
    # lm_logits = outputs

    lm_logits = torch.from_numpy(outputs[0])
    logits = lm_logits[-1, :]
    top_k, top_p, temperature = 0, 0.95, 1
    filtered_logits = top_k_top_p_filtering(
      logits, top_k=top_k, top_p=top_p, temperature=temperature
    )
    probabilities = F.softmax(filtered_logits, dim=-1)
    probabilities_logits, probabilities_position = torch.sort(
      probabilities, descending=True
    )
    predicted_token = torch.multinomial(probabilities, 1)
    output_seq = torch.cat([output_seq, predicted_token])
  output_seq = output_seq[1:-1]

  #wrzucenie sekwencji do pliku
  output_seq = [output_seq.tolist()]
  #print("output_seq", output_seq)
  output_sentence = remi_tokenizer.tokens_to_midi(output_seq)#, get_midi_programs(midi))  #self, tokens: List[int]    tokens: List[List[Union[int, List[int]]]]

  #print("output_sentence", output_sentence)
  path = "non_sentiment.mid"
  output_sentence.dump(path)

  return path #output_sentence


def inner_generation_function(generative_model, classifier, sentiment, idx2char, start_seq,
                                 seq_len=256, k=3, sent_controllers, beam_size=10):
  # Add front and end pad to the initial text
  # init_text = preprocess_sentence(init_text)
  # Empty midi to store our results
  start_seq.insert(2,0)
  midi_generated = []

  # Process initial text

  chunk_size = 250
  beams = [[] for i in range(beam_size)]
  typical_logits_wraper = TypicalLogitsWarper(mass=0.7)
  logits_penalty = RepetitionPenaltyLogitsProcessor(penalty=1.2)
  print("beams", len(beams))  # beans 4
  # model.to('cuda')
  while (len(beams[0])) < 1024:
    beams_prob = [1 for i in range(beam_size)]
    for beam in range(beam_size):
      beams[beam] = start_seq.copy()
      for i in range(chunk_size):
        with torch.no_grad():
          input_eval = torch.tensor(beams[beam])
          input_eval = torch.unsqueeze(input_eval, 0)
          # ):
          onnx_input = {"input_tokens": to_numpy(
            torch.unsqueeze(torch.Tensor(input_eval), 0))}  # [int(i) for i in lista ]), 0))} #0  279
          onnx_input['input_tokens'] = onnx_input['input_tokens'][0]

          predictions = generative_model.run(None, onnx_input)[0]
          predictions = torch.tensor(predictions)

          predictions = torch.squeeze(predictions, 0).cpu().detach().numpy()
          if predictions.shape[0] > 1:
            predictions = predictions[-1, :]
            predictions = np.expand_dims(predictions, 0)
          logits_given_prev = torch.from_numpy(predictions / 1.3)
          for s_controller in sent_controllers:
            logits_given_prev = s_controller.apply(logits_given_prev)
          filtered_logits = typical_logits_wraper(input_eval, logits_given_prev)
          filtered_logits = logits_penalty(input_eval, filtered_logits)
          log_prob_char_given_prev = F.softmax(filtered_logits[-1, :], dim=-1).detach().cpu()
          predicted_id = torch.multinomial(log_prob_char_given_prev, 1).item()
          predicted_prob = log_prob_char_given_prev[predicted_id].item()
          beams_prob[beam] *= predicted_prob
          beams[beam].append(predicted_id)
          midi_generated.append(idx2char[predicted_id])
          if len(beams[beam]) >= 1024:
            break
    stacked_classifications = []
    for beam in range(beam_size):
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
      classification_input = {"input_ids": torch.unsqueeze(torch.tensor(ids), 0),
                              "attention_mask": torch.unsqueeze(torch.tensor(attention_mask), 0)}

      classification_input = {"input_tokens": to_numpy(torch.unsqueeze(torch.tensor(ids), 0)),
                              "attention_mask": to_numpy(torch.unsqueeze(torch.tensor(attention_mask), 0))}  # [0]

      classification_output = F.softmax(
        torch.tensor(classifier.run(None, classification_input)[0])).detach().cpu().numpy()  # [0]

      stacked_classifications.append(classification_output[0])

    stacked_classifications = np.array(stacked_classifications)

    print("stacked_classifications", stacked_classifications)

    k = np.argmax(stacked_classifications[:, sentiment], axis=0)
    start_seq = beams[k]

  return start_seq



def generate_midi_with_sent(model_to_download, classifier_to_download, start_seq=[2], sentiment = 1, sent_controllers=[]):
  #GENERATIVE_MODEL
  model_path = download_metadata(model_to_download)     #, tokenizer, tokenizer_path
  ort_session = load_onnx_model(model_path)
  # output_seq = torch.tensor(start_seq)
  #CLASSIFIER
  classifier_path = download_metadata(classifier_to_download)     #, tokenizer, tokenizer_path

  clas_session = load_onnx_model(classifier_path)
  #output_seq = torch.tensor(start_seq)
  # output_seq = Tokenizer.generate_midi_from_txt(start_seq)#.encode(start_seq, return_tensors=True)
  #TOKENIZER
  pitch_range = range(21, 110)
  beat_res = {(0, 4): 8, (4, 12): 4}
  nb_velocities = 32
  additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                       'Bar': False,
                       'rest_range': (2, 8),  # (half, 8 beats)
                       'nb_tempos': 32,
                       'tempo_range': (30, 200)}  # (min, max)

  remi_tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens=True, mask=False)
  char2idx = remi_tokenizer.vocab.event_to_token
  idx2char = remi_tokenizer.vocab.token_to_event
  #
  # classification_model = MidiClassificationModule.load_from_checkpoint('test_clssifier/epoch=5-step=161-v1.ckpt')
  #start_seq = generate_midi_with_sentiment(ort_session, char2idx, idx2char, 1,  "" , SEQ_LEN, [], 10)
  output_sentence = inner_generation_function(ort_session, clas_session, sentiment, idx2char, start_seq=start_seq,
                               seq_len=SEQ_LEN, k=3, sent_controllers=sent_controllers, beam_size=10)

  #output_sentence_midi = remi_tokenizer.tokens_to_midi([start_seq])

  output_sentence = [output_sentence]#.tolist()
  #print("output_seq", output_sentence)
  output_midi = remi_tokenizer.tokens_to_midi(
    output_sentence)  # , get_midi_programs(midi))  #self, tokens: List[int]    tokens: List[List[Union[int, List[int]]]]

  #print("output_sentence", output_midi )
  path = "non_sentiment.mid"
  output_midi .dump(path)

  return path  # output_sentence
