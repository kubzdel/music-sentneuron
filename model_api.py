import base64

from fastapi import File
from pydantic import BaseModel

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
app = FastAPI()
print(os.environ["AWS_BUCKET_NAME"])
bucket_path = str(PurePosixPath('deployment', '{}'.format(model_to_download)))
print(bucket_path)
SEQ_LEN = 512


@app.get('/generate')
async def root():
    generated_file_path = generate_midi_file(model_to_download)

    def iterfile():  #
        with open(generated_file_path, mode="rb") as file_like:  #
            yield from file_like  #

    # {'message': f'Youre using, {model_to_download}',   #change filename to model name or sth
    return StreamingResponse(iterfile(), media_type="audio/midi")

class MidiDto(BaseModel):
  start_seq_file: bytes = None
  sent: int = -1


#tutaj daniel musiałby wysyłać puste argumenty jesli ich nie ma?


@app.post('/upload_file')
async def handle_file(generation_data: MidiDto):

    if (generation_data.start_seq_file != None):
        start_seq_file = base64.b64decode(generation_data.start_seq_file)
        with open("temp.mid", mode='wb') as file:
            file.write(start_seq_file)
        midi = MidiFile("temp.mid")
        tokens_start_seq = remi_tokenizer.midi_to_tokens(midi)

        if(generation_data.sent != -1):
            generated_file_path = generate_midi_with_sent(model_to_download, classifier_to_download,
                                                   start_seq=tokens_start_seq, sentiment=generation_data.sent)
        else: #if no sent
            generated_file_path = generate_midi_with_sent(model_to_download, classifier_to_download,
                                                         start_seq=tokens_start_seq)
    else:   #no start_se, is sent
        if (generation_data.sent != -1):
            generated_file_path = generate_midi_with_sent(model_to_download, classifier_to_download, sentiment=generation_data.sent)
        else:   #no start_seq, no sent
            generated_file_path = generate_midi_file(model_to_download)
    def iterfile():  #
        with open(generated_file_path, mode="rb") as file_like:  #
            yield from file_like

    return StreamingResponse(iterfile(), media_type="audio/midi")




@app.get('/inference/')
async def root(sent: int = 1, start_seq: bytes = File(...)):  # skip: int = 0, limit: int = 10
    # 150.254.131.192:8091/inference/?sent=1&start_seq=TVRoZAAAAAYAAQACAYBNVHJrAAAAEwD/UQMF3mcA/1gEBAIYCAH/LwBNVHJrAAAGigD/AxRBY291c3RpYyBHcmFuZCBQaWFubwDAAIIgkE5XYBxXAE4AgUAcAAAoTwA0W2AlVwAoAAA0AIFAJQAAJVcANU8APV9gNQAANk8APQAAPV9gNgAAO1cAPQBgIVcAJQAAOwAAQVdgQFcAQQBgIQAAJU8AMVsAQAAARFdgRAAARVdgI2sAJQAAL3cAMQAAQlsARQAAS1uBQCMAACNXAC8AgUAjAAAjV4FAIwAAKk8ANl9gJVcAKgAANgCBQCUAACpXADhPAD9fYDgAADpPAD1fAD8AAEIAAEZXAEsAYDoAAD0AYB5XACoAAEYAAEhXYEgAAElXYB4AACNPAC9fAEdXAEkAYEcAAEhXYCMAACRTAC8AADBjAEgAAEtTAFRjgRBLAABUADAkAAAkTwAwAAAwW2AkAAAwAGAkTwAwWwBKTwBRX2AkAAAnTwAwAAAzWzAnAAAzAABKAABOTwBRAABXX2AkTwAwWwBMTwBOAABUXwBXADAlTwAxWzAkAAAlAAAwAAAxAABLUwBXXzBMAABUAGApTwA1WzApAAA1ADA4TwA9XwBLAABXADAkTwAwWzA4AAA9ADAkAAAwAABITwBUXzAgTwAwW2AgAAAwADAgSwAsWzAgSwAsAAAsXwBIAABKTwBUAABWXzAgADBITwBKAABUXwBWAGAgAAAiUwAsAAAuYwBIAABIUwBUAABUY2BIAABKTwBUAABWX2AiAAAiTwAuAAAuWwBKAABKTwBWAABWX2BKAABWAGAiAAAiTwAuAAAuWwBFTwBRX2AiAAAmTwAuAAAyWwBFAABRADAmAAApTwAwWwAyAAA+TwBKXzApAAAwADAiTwAuWwA+AABGTwBKAABSXzAjTwAvWwBGADAiAAAuADBOUwBSAABaX2AjAAAjTwAvAAAvWzAjAAAvADAjTwAvW2AjAAAvAGAjTwAvW2AjAAAvAGAjTwAvW2AjAAAvAGAlawAxdwBOAABaAIFAJQAAMQCBQCVnADFzgRAlAAAxAIFwI2cAL3dgIwAAJ08ALwAAM1swJwAAMwAwI08AL1tgIwAALwBgJWcAMXMAPU8ASV9gJQAAMQAAPQAASQBgI08AL18APk8ASl9gPgAASgBgIwAAKWsALwAANXcAP1MAS2OBQCkAADUAgUApZwA1c4EQKQAANQCBcChnADR3AD8AAEFTAEsAAE1fYCgAADQAAEEAAE0AYChPADRbAEBPAExfYCgAADQAAEAAAEwAYChnADRzAEFPAE1fYCgAADQAAEEAAE0AYClnADVvAENPAE9fYEMAAE8AYCkAACprADUAADZ3AEZTAFJjgUAqAAA2AIFAKmcANnOBECoAADYAgXApZwA1dwBBUwBGAABNXwBSAGApAAA1AABBAABNAGArTwA3XwBBTwBPX2ArAAA3AABBAABPAGAqZwA2cwBCTwBOX2AqAAA2AABCAABOAGArZwA3bwBETwBQX2BEAABQAGArAAAsawA3AAA4dwBEUwBQY4FALAAAME8AOAAAPF+BQCxnADAAADhzADwAgUAsAAAwTwA4AAA8XwBDTwBEAABPXwBQAGBDAABPAGArZwAwAAA3dwA8AABBUwBNX2BBAABNADArAAA3ADBCTwBOX2BCAABOAGAvZwA7cwBCTwBOX4MAJFMALwAAMGMAOwAAQgAAQ08ATgAAT1+BQCQAACtPADAAADdbgUAkTwArAAAwWwA3AIEQJAAAMACBcB9nACt3YB8AACZPACsAADJbMCYAADIAMB9LACtbYB8AACsAYB9PACtbAEMAAE8AYB8AACsAYB9LACtfAEFPAE1fYB8AACsAAEEAAE0AYCBTACxjADxTAEhjgUAgAAAsAAAsTwA4W4FAIE8ALAAALFsAOACBQCAAACwAACxPADhfADwAAENPAEgAAE9fYEMAAE8AYCtnACwAADd3ADgAAEFTAE1fYEEAAE0AMCsAADcAMD5PAEpfYD4AAEoAYDBnADxzAEhPAFRfYEgAAFQAYEhPAFRfYEgAAFQAYCdrADAAADN3ADwAAEZTAFJjgUAnAAAuTwAzAAA6X4FALgAALmcAOgAAOnMARgAAUgCBQCtPAC4AADdfADoAAEZPAFJfYEYAAFIAYCsAACxnADcAADh3AEFTAE1fYEEAAE0AMCwAADgAMD9PAEtfYD8AAEsAYDJnAD5zAEpPAFZfYEoAAFYAYEpPAFZfYEoAAFYAYDIAAD4AAf8vAA==

    #150.254.131.192:8091/inference/?sent=1&start_seq=MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\x80MTrk\x00\x00\x00\x13\x00\xffQ\x03\x05\xdeg\x00\xffX\x04\x04\x02\x18\x08\x01\xff/\x00MTrk\x00\x00\x06\x8a\x00\xff\x03\x14Acoustic Grand Piano\x00\xc0\x00\x82 \x90NW`\x1cW\x00N\x00\x81@\x1c\x00\x00(O\x004[`%W\x00(\x00\x004\x00\x81@%\x00\x00%W\x005O\x00=_`5\x00\x006O\x00=\x00\x00=_`6\x00\x00;W\x00=\x00`!W\x00%\x00\x00;\x00\x00AW`@W\x00A\x00`!\x00\x00%O\x001[\x00@\x00\x00DW`D\x00\x00EW`#k\x00%\x00\x00/w\x001\x00\x00B[\x00E\x00\x00K[\x81@#\x00\x00#W\x00/\x00\x81@#\x00\x00#W\x81@#\x00\x00*O\x006_`%W\x00*\x00\x006\x00\x81@%\x00\x00*W\x008O\x00?_`8\x00\x00:O\x00=_\x00?\x00\x00B\x00\x00FW\x00K\x00`:\x00\x00=\x00`\x1eW\x00*\x00\x00F\x00\x00HW`H\x00\x00IW`\x1e\x00\x00#O\x00/_\x00GW\x00I\x00`G\x00\x00HW`#\x00\x00$S\x00/\x00\x000c\x00H\x00\x00KS\x00Tc\x81\x10K\x00\x00T\x000$\x00\x00$O\x000\x00\x000[`$\x00\x000\x00`$O\x000[\x00JO\x00Q_`$\x00\x00\'O\x000\x00\x003[0\'\x00\x003\x00\x00J\x00\x00NO\x00Q\x00\x00W_`$O\x000[\x00LO\x00N\x00\x00T_\x00W\x000%O\x001[0$\x00\x00%\x00\x000\x00\x001\x00\x00KS\x00W_0L\x00\x00T\x00`)O\x005[0)\x00\x005\x0008O\x00=_\x00K\x00\x00W\x000$O\x000[08\x00\x00=\x000$\x00\x000\x00\x00HO\x00T_0 O\x000[` \x00\x000\x000 K\x00,[0 K\x00,\x00\x00,_\x00H\x00\x00JO\x00T\x00\x00V_0 \x000HO\x00J\x00\x00T_\x00V\x00` \x00\x00"S\x00,\x00\x00.c\x00H\x00\x00HS\x00T\x00\x00Tc`H\x00\x00JO\x00T\x00\x00V_`"\x00\x00"O\x00.\x00\x00.[\x00J\x00\x00JO\x00V\x00\x00V_`J\x00\x00V\x00`"\x00\x00"O\x00.\x00\x00.[\x00EO\x00Q_`"\x00\x00&O\x00.\x00\x002[\x00E\x00\x00Q\x000&\x00\x00)O\x000[\x002\x00\x00>O\x00J_0)\x00\x000\x000"O\x00.[\x00>\x00\x00FO\x00J\x00\x00R_0#O\x00/[\x00F\x000"\x00\x00.\x000NS\x00R\x00\x00Z_`#\x00\x00#O\x00/\x00\x00/[0#\x00\x00/\x000#O\x00/[`#\x00\x00/\x00`#O\x00/[`#\x00\x00/\x00`#O\x00/[`#\x00\x00/\x00`%k\x001w\x00N\x00\x00Z\x00\x81@%\x00\x001\x00\x81@%g\x001s\x81\x10%\x00\x001\x00\x81p#g\x00/w`#\x00\x00\'O\x00/\x00\x003[0\'\x00\x003\x000#O\x00/[`#\x00\x00/\x00`%g\x001s\x00=O\x00I_`%\x00\x001\x00\x00=\x00\x00I\x00`#O\x00/_\x00>O\x00J_`>\x00\x00J\x00`#\x00\x00)k\x00/\x00\x005w\x00?S\x00Kc\x81@)\x00\x005\x00\x81@)g\x005s\x81\x10)\x00\x005\x00\x81p(g\x004w\x00?\x00\x00AS\x00K\x00\x00M_`(\x00\x004\x00\x00A\x00\x00M\x00`(O\x004[\x00@O\x00L_`(\x00\x004\x00\x00@\x00\x00L\x00`(g\x004s\x00AO\x00M_`(\x00\x004\x00\x00A\x00\x00M\x00`)g\x005o\x00CO\x00O_`C\x00\x00O\x00`)\x00\x00*k\x005\x00\x006w\x00FS\x00Rc\x81@*\x00\x006\x00\x81@*g\x006s\x81\x10*\x00\x006\x00\x81p)g\x005w\x00AS\x00F\x00\x00M_\x00R\x00`)\x00\x005\x00\x00A\x00\x00M\x00`+O\x007_\x00AO\x00O_`+\x00\x007\x00\x00A\x00\x00O\x00`*g\x006s\x00BO\x00N_`*\x00\x006\x00\x00B\x00\x00N\x00`+g\x007o\x00DO\x00P_`D\x00\x00P\x00`+\x00\x00,k\x007\x00\x008w\x00DS\x00Pc\x81@,\x00\x000O\x008\x00\x00<_\x81@,g\x000\x00\x008s\x00<\x00\x81@,\x00\x000O\x008\x00\x00<_\x00CO\x00D\x00\x00O_\x00P\x00`C\x00\x00O\x00`+g\x000\x00\x007w\x00<\x00\x00AS\x00M_`A\x00\x00M\x000+\x00\x007\x000BO\x00N_`B\x00\x00N\x00`/g\x00;s\x00BO\x00N_\x83\x00$S\x00/\x00\x000c\x00;\x00\x00B\x00\x00CO\x00N\x00\x00O_\x81@$\x00\x00+O\x000\x00\x007[\x81@$O\x00+\x00\x000[\x007\x00\x81\x10$\x00\x000\x00\x81p\x1fg\x00+w`\x1f\x00\x00&O\x00+\x00\x002[0&\x00\x002\x000\x1fK\x00+[`\x1f\x00\x00+\x00`\x1fO\x00+[\x00C\x00\x00O\x00`\x1f\x00\x00+\x00`\x1fK\x00+_\x00AO\x00M_`\x1f\x00\x00+\x00\x00A\x00\x00M\x00` S\x00,c\x00<S\x00Hc\x81@ \x00\x00,\x00\x00,O\x008[\x81@ O\x00,\x00\x00,[\x008\x00\x81@ \x00\x00,\x00\x00,O\x008_\x00<\x00\x00CO\x00H\x00\x00O_`C\x00\x00O\x00`+g\x00,\x00\x007w\x008\x00\x00AS\x00M_`A\x00\x00M\x000+\x00\x007\x000>O\x00J_`>\x00\x00J\x00`0g\x00<s\x00HO\x00T_`H\x00\x00T\x00`HO\x00T_`H\x00\x00T\x00`\'k\x000\x00\x003w\x00<\x00\x00FS\x00Rc\x81@\'\x00\x00.O\x003\x00\x00:_\x81@.\x00\x00.g\x00:\x00\x00:s\x00F\x00\x00R\x00\x81@+O\x00.\x00\x007_\x00:\x00\x00FO\x00R_`F\x00\x00R\x00`+\x00\x00,g\x007\x00\x008w\x00AS\x00M_`A\x00\x00M\x000,\x00\x008\x000?O\x00K_`?\x00\x00K\x00`2g\x00>s\x00JO\x00V_`J\x00\x00V\x00`JO\x00V_`J\x00\x00V\x00`2\x00\x00>\x00\x01\xff/\x00
    #150.254.131.192:8091/inference/?sent=1&start_seq=MThd\x00\x00\x00\x06\x00\x01\x00\x02\x01\x80MTrk\x00\x00\x00\x13\x00\xffQ\x03\x04\x93\xe0\x00\xffX\x04\x04\x02\x18\x08\x01\xff/\x00MTrk\x00\x00\x05\xdc\x00\xff\x03\x14Acoustic Grand Piano\x00\xc0\x00\x00\x906_\x00?[\x83\x00*S\x006\x00\x009_\x81@?\x000*\x0006_\x009\x00\x00?_`-W\x006\x00\x83\x00,_\x00-\x00\x00;_\x00?\x00\x81p;\x00\x00@_\x81\x10,\x00\x00-_\x00@\x00\x00D_\x00L_\x81p-\x00\x004_\x81\x10,_\x004\x00\x83\x00,\x00\x00-_\x83\x00,_\x00-\x00\x00D\x00\x00G_\x00L\x00\x81pG\x00\x00H_\x81\x10,\x00\x00-_\x00H\x00\x00K_\x81p-\x0004_`,_\x004\x00\x81p,\x0004_`-_\x004\x00\x81p;_\x81\x10-\x00\x00/_\x00;\x00\x00K\x00\x81p/\x00\x004_\x81\x10-_\x004\x00\x00Q_\x81p-\x00\x004_\x81\x10,_\x004\x00\x83\x00,\x00\x00-_\x00P_\x00Q\x00\x81\x10P\x00\x81p,_\x00-\x00\x00Q_\x81pK_\x00Q\x00\x81\x10,\x00\x00-_\x81p-\x00\x004_\x81\x10,_\x004\x00\x81p,\x0004_`-_\x004\x00\x00K\x00\x00O_\x81\x10O\x00\x81p,_\x00-\x00\x00P_\x81p,\x00\x00,_\x81\x10,\x00\x00-_\x00I_\x00L_\x00P\x00\x83\x00,_\x00-\x00\x83\x00,\x00\x00-_\x83\x00,_\x00-\x00\x00E_\x00I\x00\x00L\x00\x81pE\x00\x00G_\x81\x10,\x00\x00-_\x00G\x00\x00K_\x81p-\x00\x004_04_`,_\x004\x00\x004\x00\x83\x00,\x00\x00-_\x81p@_\x81\x10,_\x00-\x00\x00@\x00\x00@_\x00K\x00\x81p@\x00\x00@_\x81\x10,\x00\x00-_\x00@\x00\x00C_\x00I_\x81p-\x00\x004_\x81\x10,_\x004\x00\x81p,\x0004_`-_\x004\x00\x83\x00,_\x00-\x00\x00C\x00\x00I\x00\x00K_\x81\x10I_\x00K\x00`I\x00\x00J_\x81\x10,\x00\x00-_\x00J\x00\x00L_\x81p-\x00\x004_\x81\x10,_\x004\x00\x83\x00,\x00\x00-_\x81p@_\x81\x10,_\x00-\x00\x00@\x00\x00K_\x00L\x00\x81\x10I_\x00K\x00`G_\x00I\x00\x81\x10,\x00\x00-_\x00G\x00\x81p-\x0004_`,_\x004\x00\x81p,\x0004_`-_\x004\x00\x81p;_\x81\x10-\x00\x00/_\x00;\x00\x00Q_\x81p/\x00\x004_\x81\x10-_\x004\x00\x00Q\x00\x00U_\x81p4_\x81\x10,_\x00-\x00\x004\x00\x83\x00,\x00\x00-_\x00U\x00\x00X_\x83\x00,_\x00-\x00\x81pL_\x81\x10,\x00\x00-_\x00L\x00\x00N_\x00X\x00\x81p-\x00\x004_\x81\x10,_\x004\x00\x81p,\x0004_`-_\x004\x00\x83\x00,_\x00-\x00\x00N\x00\x81pL_\x81\x10,\x00\x00-_\x00L\x00\x00Q_\x81p4_\x81\x10,_\x00-\x00\x004\x00\x83\x00,\x00\x00-_\x00P_\x00Q\x00\x83\x00,_\x00-\x00\x81pQ_\x81\x10*_\x00,\x00\x006_\x00I_\x00P\x00\x00Q\x00\x83\x00*\x00\x00*_\x006\x00\x83\x00*\x00\x00*_\x00G_\x00I\x00\x83\x00*\x00\x00*_\x83\x00*\x00\x00/_\x00G\x00\x00N_\x81\x10K_`6_\x00K\x0006_`/\x00\x00/_\x006\x00\x006\x00\x81p/\x0006_`/_\x006\x00\x00B_\x81p6_\x81\x10/\x00\x00/_\x006\x00\x00N\x00\x81pB\x00\x00B_\x81\x10/\x00\x00/_\x00?_\x00B\x00\x81p/\x00\x006_06_`/_\x006\x00\x006\x00\x81@B_0/\x00\x00?\x0006_\x00F_`/_\x006\x000B\x00\x82P/\x00\x00/_\x00F\x00\x81p/\x00\x00/_\x81\x10/\x00\x001_\x008_\x00I_\x81p1\x00\x008\x0008_`1_\x008\x00\x81p1\x0008_`1_\x008\x00\x81p1\x00\x008_\x81\x101_\x008\x00\x00I\x00\x81p1\x00\x00=_\x81\x102_\x006_\x00=\x00\x00?_\x81p2\x00\x006\x00\x00?\x0006_\x00@_`/_\x006\x00\x00?_\x00@\x00\x00K_\x81p/\x00\x00?\x00\x00K\x0006_\x00E_`/_\x006\x00\x00E\x00\x81p/\x00\x006_\x00B_\x81\x10/_\x006\x00\x00@_\x00B\x00\x00I_\x81p/\x00\x00@\x00\x00B_\x00I\x00\x00J_\x81\x10/_\x00B\x00\x00B_\x00I_\x00J\x00\x81p/\x00\x006_\x81\x10/_\x006\x00\x81p/\x0006_`/_\x006\x00\x00B\x00\x83\x00/\x00\x00/_\x00I\x00\x00N_\x83\x00/\x00\x00/_\x00B_\x00K_\x00N\x00\x83\x00/\x00\x00/_\x83\x00/\x00\x00/_\x00B\x00\x83\x00/\x00\x00/_\x83\x00/\x00\x00/_\x00N_\x81p/\x00\x006_\x81\x10/_\x006\x00\x83\x00/\x00\x00/_\x83\x00/\x00\x00/_\x00K\x00\x00K_\x00N\x00\x81pI_\x00K\x00\x81\x10/\x00\x006_\x00I\x00\x00K_\x83\x00/_\x006\x00\x83\x00/\x00\x00/_\x83\x00/\x00\x00/_\x00K\x00\x00K_\x81pI_\x00K\x00\x81\x10/\x00\x00/_\x006_\x00I\x00\x00K_\x81p/\x00\x006\x00\x006_\x81\x10/_\x006\x00\x81p/\x0006_`/_\x006\x00\x00B_\x00K\x00\x83\x00/\x00\x00/_\x81p/\x00\x00B\x00\x00B_\x81\x101_\x00B\x00\x00D_\x00L_\x81p1\x00\x008_\x81\x101_\x008\x00\x81p1\x0008_`8\x00\x83\x00D\x00\x00L\x00\x01\xff/\x00'

    if (sent == 0 or sent == 1):
        if (start_seq != ''):  # start_seq=[2]    weź start seq z @app.post /upload_file
            print("sent & start_seq")

            print("1")
            midi_seq = base64.b64decode(start_seq)# + b'=' * (-len(start_seq) % 4))#, errors='replace')
            print("2")
            #return {'message': midi_seq}#f'start_seq'}


            tokens_start_seq = remi_tokenizer.midi_to_tokens(midi_seq)#start_seq)#midi_seq)
            #return {'message': f'{tokens_start_seq}'}        #return {'message': f'Dont be afraid, this API still works {name}'}
            midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download,
                                                         start_seq=tokens_start_seq, sentiment=sent)
        else:
            print("no start_seq")
            midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download, sentiment=sent)
    else:
        print("non-sentimental")
        midi_with_sent_txt = generate_midi_file(model_to_download)

    def iterfile():  #
        with open(midi_with_sent_txt, mode="rb") as file_like:  #
            yield from file_like  #

    return StreamingResponse(iterfile(), media_type="audio/midi")  # a #None #


@app.get('/{name}')
def abc(name: str=''):#sent: int = 1):
    #150.254.131.192:8091/abc/?sent=1
    #midi_with_sent_txt = generate_midi_with_sent(model_to_download, classifier_to_download, sentiment=sent)
    return {'message': f'Hello {name}'}

@app.post('/change_model')
def change_model():
    return {'message': f'change model'}


if __name__ == "__main__":

    uvicorn.run("model_api:app", host="0.0.0.0", port=8091  # 8083
                , reload=True)


