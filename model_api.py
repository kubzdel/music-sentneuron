from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
# import uvicorn
# import os
# from pathlib import PurePosixPath
# import torch
# from minio import Minio
# from starlette.responses import StreamingResponse
# from cachetools.func import lru_cache
#
# import onnxruntime
# import torch.nn.functional as F
# from tokenizer import MidiTokenizer
# import music21 as m21
# from utils import top_k_top_p_filtering

from helper_functions_org import *
from dotenv import load_dotenv
load_dotenv()

from joblib import dump


#TOKENIZER (to onnx file)
pitch_range = range(21, 110)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'Bar': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,
                     'tempo_range': (30, 200)}  # (min, max)

remi_tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens=True, mask=False)



#MODEL_NAME = "model123"   #env - który model z bucketa chcemy wczytać
model_to_download = "model_remi.onnx"
classifier_to_download = "classifier.onnx"
#declaring FastAPI instance
app = FastAPI()
print(os.environ["AWS_BUCKET_NAME"])
bucket_path = str(PurePosixPath('deployment','{}'.format(model_to_download )))
print(bucket_path )
SEQ_LEN = 512



# @app.get('/')
# async def root():
#   generated_file_path = generate_midi_file()
#   with open(generated_file_path, "rb") as midi_bytes:
#     data = midi_bytes.read()
#     return StreamingResponse(data, media_type="audio/midi")

@app.get('/generate')
async def root():
  generated_file_path = generate_midi_file(model_to_download)
  def iterfile():  #
    with open(generated_file_path, mode="rb") as file_like:  #
      yield from file_like  #
  #{'message': f'Youre using, {model_to_download}',   #change filename to model name or sth
  return StreamingResponse(iterfile(), media_type="audio/midi")

@app.get('/sent{senti}')
async def root(senti: int = 1, start_seq: str = ''):  #skip: int = 0, limit: int = 10
  #150.254.131.192:8080/sent0?start_seq="sekwencja startowa"
  if(senti==0 or senti==1):
    if(start_seq != ''):                              #start_seq=[2]    weś start seq z @app.post /upload_file
      midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download, start_seq = start_seq, sentiment=senti)
      a="start_seq neg"
    else:
      midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download, sentiment=senti)
      a = "neg"
  else:
    message = "Sentyment może przyjmować wartości 0 (neg) lub 1 (poz). Podaj poprawny sentyment"
    return message
  #midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download, sentiment= sent)
  def iterfile():  #
    with open(midi_with_sent_txt, mode="rb") as file_like:  #
      yield from file_like  #

  # {'message': f'Youre using, {model_to_download}',   #change filename to model name or sth
  return StreamingResponse(iterfile(), media_type="audio/midi") #a #None #

@app.post('/upload_file')
async def handle_file(start_seq_file: UploadFile = File(...)):
  print("im in upload file")
  print(start_seq_file.filename)
  midi_start_seq = await start_seq_file.read()
  print(midi_start_seq)
  tokens_start_seq = remi_tokenizer.midi_to_tokens(midi_start_seq)
  print("got it", tokens_start_seq)
  pass

@app.post('/change_model')
def change_model():
  pass


@app.get('/{name}')#, response_class=HTMLResponse)
def hello_name(name: str):
  # func that takes only str as input and outputs following message:
  return {'message': f'Dont be afraid, this API still works {name}'}
#templates.TemplateResponse("basic_front_end.html ")

if __name__ == "__main__":
  uvicorn.run("model_api:app", host="0.0.0.0", port=8080, reload=True)
