import os
from pathlib import PurePosixPath

import torch
from minio import Minio
from transformer_generative import MidiTrainingModule
from dotenv import load_dotenv
from datetime import datetime

# returns current date and time
now = str(datetime.now())
now = now.replace(":", "").replace(".", "").replace(" ", "_")
model_file_path = "model_remi_"+now+".onnx"  #+days+" "+mins+".onnx"
model_checkpoint_file = '/home/dsolarska/music-sentneuron/API remi model/last.ckpt'

load_dotenv()
# print(os.environ)

SEQ_LEN = 512
my_lightning_module = MidiTrainingModule.load_from_checkpoint(
    model_checkpoint_file, n_head=6, n_embd=402, n_layer=6, batch_size=16,
    epochs=50, seq_len=1024, tokenizer=None, training_stride=1024, validation_stride=1024, samples_count=8725,
    n_units=4096, vocab_size=279)
model_to_export = my_lightning_module.model  # --n_embd 768 --n_layer 4 --batch 32 --epochs 5 --seq_len 512 --validation_stride 512 --training_stride 512 --n_head 4         |
model_to_export.eval()
sample_input_for_onnx = torch.randint(high=1, size=(1, SEQ_LEN))
dynamic_axes = {"input_tokens": [1], "logits": [1]}

model_to_export.to("cuda")
torch.onnx.export(model_to_export,
                  export_params=True,
                  opset_version=11,
                  args=sample_input_for_onnx,
                  f=model_file_path,
                  dynamic_axes=dynamic_axes,
                  input_names=["input_tokens"],
                  output_names=["logits"])

MINIO_API_HOST = "http://150.254.131.193:9000"  # "http://localhost:9000"
client = Minio("150.254.131.193:9000", access_key=os.environ["AWS_ACCESS_KEY_ID"],
               secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], secure=False)
bucket_path = str(PurePosixPath('deployment', '{}'.format(model_file_path)))  # aws_path +tokenizer
client.fput_object(os.environ["AWS_BUCKET_NAME"], bucket_path, model_file_path)  # aws_bucket
# bucket_path = str(PurePosixPath('deployment','{}'.format("char2idx.json")))      #aws_path +tokenizer
# client.fput_object(os.environ["AWS_BUCKET_NAME"], bucket_path, "trained/char2idx.json")        #aws_bucket

