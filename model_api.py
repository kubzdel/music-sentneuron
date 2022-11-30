import base64

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from helper_functions_org import *
from dotenv import load_dotenv
load_dotenv()


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



@app.get('/inference')
async def root(sent: int = 1, start_seq: bytes = ''):  #skip: int = 0, limit: int = 10
  #150.254.131.192:8081/inference?sent=1?start_seq="sekwencja startowa"
  if(sent==0 or sent==1):
    if(start_seq != ''):                              #start_seq=[2]    weś start seq z @app.post /upload_file
      print("sent & start_seq")
      midi_seq = base64.b64decode(start_seq)
      tokens_start_seq = remi_tokenizer.midi_to_tokens(midi_seq)
      midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download, start_seq = tokens_start_seq, sentiment=sent)
    else:
      print("no start_seq")
      midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download, sentiment=sent)
  else:
    print("non-sentimental")
    midi_with_sent_txt = generate_midi_file(model_to_download)

  def iterfile():  #
    with open(midi_with_sent_txt, mode="rb") as file_like:  #
      yield from file_like  #

  return StreamingResponse(iterfile(), media_type="audio/midi") #a #None #



@app.get('/{name}')#, response_class=HTMLResponse)
def hello_name(name: str):
  # func that takes only str as input and outputs following message:
  return {'message': f'Dont be afraid, this API still works {name}'}
#templates.TemplateResponse("basic_front_end.html ")

if __name__ == "__main__":
  uvicorn.run("model_api:app", host="0.0.0.0", port=8081  #8083
              , reload=True)


