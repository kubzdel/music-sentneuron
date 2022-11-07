from helper_functions_api import *
from dotenv import load_dotenv
load_dotenv()
from joblib import dump

#MODEL_NAME = "model123"   #env - który model z bucketa chcemy wczytać
model_to_download = "model_remi.onnx"
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

@app.get('/sent')
async def root():
  midi_with_sent_txt = generate_midi_with_sent(model_to_download)
  def iterfile():  #
    with open(midi_with_sent_txt, mode="rb") as file_like:  #
      yield from file_like  #

  # {'message': f'Youre using, {model_to_download}',   #change filename to model name or sth
  return StreamingResponse(iterfile(), media_type="audio/midi")

@app.post('/change_model')
def change_model():
  pass

@app.get('/{name}')
def hello_name(name: str):
  # func that takes only str as input and outputs following message:
  return {'message': f'Weltyrtcome , {name}'}

if __name__ == "__main__":
  uvicorn.run("model_api:app", host="0.0.0.0", port=8083, reload=True)
