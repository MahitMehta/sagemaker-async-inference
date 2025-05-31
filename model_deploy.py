import os
import time
import boto3
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch
import shutil
from constants import *
import libarchive

def create_tar_gz_with_libarchive(output_dir, tar_file):
    start = time.time()

    with libarchive.file_writer(tar_file, 'gnutar', 'gzip') as archive:
        archive.add_files(output_dir)

    print(f"Model archive created at {tar_file} in {time.time() - start:.2f} seconds.")

tmp_dir = "./tmp"
os.makedirs(tmp_dir, exist_ok=True)

output_dir = os.path.join(tmp_dir, "sdxl_model")
tar_file = os.path.join(tmp_dir, "model.tar.gz")
local_inference_file = "./code/inference.py" 

if not os.path.exists(output_dir):
    print("Downloading model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline.save_pretrained(output_dir)
else:
    print(f"Model already exists at {output_dir}, skipping download.")

model_code_dir = os.path.join(output_dir, "code")
os.makedirs(model_code_dir, exist_ok=True)

print(f"Copying {local_inference_file} to {model_code_dir}/inference.py...")
shutil.copy(local_inference_file, os.path.join(model_code_dir, "inference.py"))

print("Compressing into model.tar.gz...")
create_tar_gz_with_libarchive(output_dir, tar_file)

print(f"Uploading {tar_file} to s3://{S3_PROJECT_BUCKET}/{MODEL_S3_KEY}...")
s3 = boto3.client("s3")
s3.upload_file(tar_file, S3_PROJECT_BUCKET, MODEL_S3_KEY)

print("Model Uploaded successfully to S3.")
