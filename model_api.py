import base64
import uvicorn
from sentiment_controllers import TempoController, PitchController
from fastapi import FastAPI #, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from helper_functions_org import *
from dotenv import load_dotenv

from miditoolkit import MidiFile


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

# MODEL_NAME = "model123"   #env - który model z bucketa chcemy wczytać
model_to_download = "model_remi.onnx"
classifier_to_download = "classifier.onnx"
# declaring FastAPI instance
app = FastAPI(middleware=[
    Middleware(CORSMiddleware, allow_origins=["*"])
])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(os.environ["AWS_BUCKET_NAME"])
bucket_path = str(PurePosixPath('deployment', '{}'.format(model_to_download)))
print(bucket_path)
SEQ_LEN = 512

class MidiDto(BaseModel):
  start_seq_file: bytes = None
  sent: int = -1
  tempo: str = ""
  pitch: str = ""   #sounds_range



@app.post('/generate')
async def handle_file(generation_data: MidiDto):
    sent_controllers = []
    if (generation_data.tempo != ""):
        sent_controllers.append(TempoController(mode=generation_data.tempo))
    if (generation_data.pitch != ""):
        sent_controllers.append(PitchController(mode=generation_data.pitch))


    if (generation_data.start_seq_file != None):
        start_seq_file = base64.b64decode(generation_data.start_seq_file)
        with open("temp.mid", mode='wb') as file:
            file.write(start_seq_file)
        midi = MidiFile("temp.mid")
        tokens_start_seq = remi_tokenizer.midi_to_tokens(midi)[0]
        print(tokens_start_seq)


        if(generation_data.sent != -1):
            generated_file_path = generate_midi_with_sent(model_to_download, classifier_to_download,
                                                   start_seq=tokens_start_seq, sentiment=generation_data.sent, sent_controllers = sent_controllers)
        else: #if no sent
            generated_file_path = generate_midi_with_sent(model_to_download, classifier_to_download,
                                                         start_seq=tokens_start_seq, sent_controllers = sent_controllers)
    else:   #no start_se, is sent
        if (generation_data.sent != -1):
            generated_file_path = generate_midi_with_sent(model_to_download, classifier_to_download, sentiment=generation_data.sent, sent_controllers = sent_controllers)
        else:   #no start_seq, no sent
            generated_file_path = generate_midi_file(model_to_download)
    def iterfile():  #
        with open(generated_file_path, mode="rb") as file_like:  #
            yield from file_like
    return StreamingResponse(iterfile(), media_type="audio/midi")


@app.post('/change_model')
def change_model():
    return {'message': f'change model'}


if __name__ == "__main__":

    uvicorn.run("model_api:app", host="0.0.0.0", port=8092  # 8083
                , reload=True)
