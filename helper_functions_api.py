from fastapi import FastAPI
import uvicorn
import os
from pathlib import PurePosixPath
import torch
from minio import Minio
from starlette.responses import StreamingResponse
from cachetools.func import lru_cache

import onnxruntime
import torch.nn.functional as F
from tokenizer import MidiTokenizer
import music21 as m21
from utils import top_k_top_p_filtering



def encoding2midi(note_encoding, ts_duration=0.25):
  notes = []

  velocity = 100
  duration = "16th"
  dots = 0

  ts = 0
  for note in note_encoding.split(" "):
    if len(note) == 0:
      continue

    elif note[0] == "w":
      wait_count = int(note.split("_")[1])
      ts += wait_count

    elif note[0] == "n":
      pitch = int(note.split("_")[1])
      note = m21.note.Note(pitch)
      note.duration = m21.duration.Duration(type=duration, dots=dots)
      note.offset = ts * ts_duration
      note.volume.velocity = velocity
      notes.append(note)

    elif note[0] == "d":
      duration = note.split("_")[1]
      dots = int(note.split("_")[2])

    elif note[0] == "v":
      velocity = int(note.split("_")[1])

    elif note[0] == "t":
      tempo = int(note.split("_")[1])

      if tempo > 0:
        mark = m21.tempo.MetronomeMark(number=tempo)
        mark.offset = ts * ts_duration
        notes.append(mark)

  piano = m21.instrument.fromString("Piano")
  notes.insert(0, piano)

  piano_stream = m21.stream.Stream(notes)
  main_stream = m21.stream.Stream([piano_stream])

  return m21.midi.translate.streamToMidiFile(main_stream)

def write(encoded_midi, path):
  # Base class checks if output path exists
  midi = encoding2midi(encoded_midi)
  midi.open(path, "wb")
  midi.write()
  midi.close()

def to_numpy(tensor):
  return tensor.detach().numpy()




@lru_cache
def download_metadata():
  #MODEL_NAME = os.getenv('MODEL_NAME')  # -środowiskowa
  # odwortne do export model
  #download model and tokenizer then load locally
  client = Minio("150.254.131.193:9000", access_key=os.environ["AWS_ACCESS_KEY_ID"],
                 secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], secure=False)
  model_bucket_path = str(PurePosixPath('deployment', '{}'.format(model_to_download)))  # aws_path

  model = client.fget_object(os.environ["AWS_BUCKET_NAME"], model_bucket_path, "downloaded_model.onnx")

  token_bucket_path = str(PurePosixPath('deployment', '{}'.format("char2idx.json")))
  tokenizer_file = client.fget_object(os.environ["AWS_BUCKET_NAME"], token_bucket_path, "downloaded_tokenizer.json")

  return model, "downloaded_model.onnx", tokenizer_file, "downloaded_tokenizer.json"

def inference_model(onnxsession, tokenizer, start_seq = "\n"):
  output_seq = tokenizer.encode(start_seq, return_tensors=True)
  predicted_token = torch.Tensor([1])
  while (
          len(output_seq) < SEQ_LEN - 1 and predicted_token.unsqueeze(-1)[0] != 0
  ):
    onnx_input = {"input_tokens": to_numpy(torch.unsqueeze(output_seq, 0))}
    outputs = onnxsession.run(None, onnx_input)[0]

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
  output_sentence = tokenizer.decode(output_seq)
  return output_sentence

def generate_midi_file():
  #wczytanie modelu z bucketa (potrzebny model i tokenizer (dane na buckecie)
  model, model_path, tokenizer, tokenizer_path = download_metadata()
  tokenizer = MidiTokenizer(from_file=tokenizer_path) #'trained/char2idx.json')

  # wczytanie modelu
  ort_session = onnxruntime.InferenceSession(model_path)
  midi_events = inference_model(ort_session, tokenizer)

  path = "tmp.mid"
  write(midi_events, path)

  return path

