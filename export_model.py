import os.path
from pathlib import PurePosixPath

import boto3
from dotenv import load_dotenv

# load_dotenv()
# model_file_path = "classification_img_model_2022_07_22.onnx"
# s3_client = boto3.client('s3')
# aws_path = f"dev/image_embeddings/{model_file_path}"
# s3_client.upload_file(model_file_path, "tise-ml", aws_path)
from minio import Minio

load_dotenv()
model_name = "README.md"
MINIO_API_HOST = "http://localhost:9000"
client = Minio("localhost:9000", access_key=os.environ["AWS_ACCESS_KEY_ID"], secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], secure=False)
aws_path = str(PurePosixPath('deployment','{}'.format(model_name)))
client.fput_object(os.environ["AWS_BUCKET_NAME"], aws_path, model_name)
client.get