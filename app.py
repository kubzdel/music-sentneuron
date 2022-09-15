import joblib
from fastapi import FastAPI
import uvicorn

import os
from pathlib import PurePosixPath
import torch
from minio import Minio
from transformer_generative import MidiTrainingModule
from cachetools.func import lru_cache
import midi_generator
import json
from dotenv import load_dotenv
load_dotenv()
import onnx
import onnxruntime
# use joblib library to to the exporting
from tokenizer import MidiTokenizer
import music21 as m21

from joblib import dump

#MODEL_NAME = "model123"   #env - który model z bucketa chcemy wczytać
model_to_download = "model.onnx"
#declaring FastAPI instance
app = FastAPI()
#print(os.environ["AWS_BUCKET_NAME"])
bucket_path = str(PurePosixPath('deployment','{}'.format(model_to_download )))
print(bucket_path )
SEQ_LEN = 512


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
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

sample_input_for_onnx = torch.randint(high=1, size=(1, SEQ_LEN))
#print("tensor")
#print(sample_input_for_onnx)

@lru_cache
def download_metadata():
  #MODEL_NAME = os.getenv('MODEL_NAME')  # -środowiskowa
  # odwortne do export model
  #download model and tokenizer then load locally
  client = Minio("150.254.131.193:9000", access_key=os.environ["AWS_ACCESS_KEY_ID"],
                 secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], secure=False)
  model_bucket_path = str(PurePosixPath('deployment', '{}'.format(model_to_download)))  # aws_path
  #model = client.get_object(os.environ["AWS_BUCKET_NAME"], model_bucket_path)
  model = client.fget_object(os.environ["AWS_BUCKET_NAME"], model_bucket_path, "downloaded_model.onnx")



  token_bucket_path = str(PurePosixPath('deployment', '{}'.format("char2idx.json")))
  #tokenizer = client.get_object(os.environ["AWS_BUCKET_NAME"], token_bucket_path)
  tokenizer_file = client.fget_object(os.environ["AWS_BUCKET_NAME"], token_bucket_path, "downloaded_tokenizer.json")
  #loaded_tokenizer = joblib.load("downloaded_tokenizer.json")
  #open('downloaded_tokenizer.json', 'access_mode')
  #joblib.dump(tokenizer, 'downloaded_tokenizer.json')
  # tu pobiera tokenizer i model ^ na początek mogę uprościć to do jednego pliku - np sam model

  #print(model.data) #model.data.decode()

  return "downloaded_model.onnx", tokenizer_file

def generate_midi_file(step) -> dict:
  #wczytanie modelu z bucketa (potrzebny model i tokenizer (dane na buckecie
  model, tokenizer_file = download_metadata()
  tokenizer = MidiTokenizer(from_file=tokenizer_file) #'trained/char2idx.json')

  # wczytanie modelu
  onnx_model = onnx.load(model)  # "downloaded_model.onnx")
  onnx.checker.check_model(onnx_model)
  ort_session = onnxruntime.InferenceSession(model)

  # compute ONNX Runtime output prediction
  # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(sample_input_for_onnx)}
  # ort_outs = ort_session.run(None, ort_inputs)
  # output_song = ort_outs[0]
  # print("output_song?\n")
  # print(output_song)


  """ Predict function.
  :param sample: dictionary with the text we want to classify.
  Returns:
      Dictionary with the input text and the predicted label.
  """

  onnx_model.eval()
  onnx_model.to('cpu')
  predicted_token = torch.Tensor([1])

  with torch.no_grad():
    output_seq = tokenizer.encode('\n', return_tensors=True)
    while (
            len(output_seq) < SEQ_LEN - 1 and predicted_token.unsqueeze(-1)[0] != 0
    ):
      outputs = ort_session.run(None, to_numpy(sample_input_for_onnx))  # self.forward(output_seq, permute=False)
      lm_logits = outputs
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
    print(output_sentence)
    #Path(f'{self.run_name}_generated_samples').mkdir(exist_ok=True)
    #midi_path = f'{self.run_name}_generated_samples/{self.run_name}_step_{str(step)}.mid'
    write(output_sentence, "generated_to_upload.mid")
    #self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=midi_path)
  onnx_model.cuda()
  onnx_model.train()
  return output_sentence




@app.get('/')
async def root():
  model, tokenizer = download_metadata()
  generate_midi_file(2)
  print(model)
  print(tokenizer)
  #generate_midi_file()
  return {tokenizer}

@app.post('/change_model')
def change_model():
  pass

@app.get('/{name}')
def hello_name(name: str):
  #func that takes only str as input and outputs following message:
  return {'message': f'Welcome , {name}'}

if __name__ == "__main__":
  uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
