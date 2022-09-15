import os
from pathlib import PurePosixPath

import torch
from minio import Minio
from transformer_generative import MidiTrainingModule
from dotenv import load_dotenv
load_dotenv()
#print(os.environ)

SEQ_LEN = 512
#my_lightning_module = MidiTrainingModule('path to my checkpoint',seq_len = 256)
#tokenizer = PreTrainedTokenizerFast(tokenizer_file='')
#vocab_size = len(tokenizer)
my_lightning_module = MidiTrainingModule.load_from_checkpoint('trained_model/model_for_export/epoch=1-step=12744.ckpt',n_head=4, n_embd=768, n_layer=4, batch_size=32, epochs=5, seq_len=512, tokenizer=None, training_stride=512 ,validation_stride=512, samples_count=203952, n_units=4096, vocab_size=342)
model_to_export = my_lightning_module.model                                                                                #--n_embd 768 --n_layer 4 --batch 32 --epochs 5 --seq_len 512 --validation_stride 512 --training_stride 512 --n_head 4         |
model_to_export.eval()
sample_input_for_onnx = torch.randint(high=1, size=(1, SEQ_LEN))
dynamic_axes = {"input_tokens": [1], "logits": [1]}

# model_folder_path = os.path.join(self._artifacts_dir_path, self._stage_name)
# Path(model_folder_path).mkdir(parents=True, exist_ok=True)
# model_file_path = os.path.join(model_folder_path, '{}.onnx'.format(run_name))
torch.onnx.export(model_to_export,
                  export_params=True,
                  opset_version=11,
                  args = sample_input_for_onnx,
                  f= "model.onnx",  #model_file_path
                  dynamic_axes=dynamic_axes,
                  input_names=["input_tokens"],
                  output_names=["logits"])

model_file_path = "model.onnx"     #
MINIO_API_HOST = "http://150.254.131.193:9000" #"http://localhost:9000"
client = Minio("150.254.131.193:9000", access_key=os.environ["AWS_ACCESS_KEY_ID"], secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], secure=False)
bucket_path = str(PurePosixPath('deployment','{}'.format(model_file_path)))      #aws_path +tokenizer
client.fput_object(os.environ["AWS_BUCKET_NAME"], bucket_path, model_file_path)        #aws_bucket
bucket_path = str(PurePosixPath('deployment','{}'.format("char2idx.json")))      #aws_path +tokenizer
client.fput_object(os.environ["AWS_BUCKET_NAME"], bucket_path, "trained/char2idx.json")        #aws_bucket

#client.get_object("MODELS", MODEL_NAME)
#model = client.fget_object("AWS_BUCKET_NAME", model_to_download, str(PurePosixPath('deployment','{}'.format(model_to_download))))
