import os
import tarfile
import boto3
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch
import shutil
from constants import *

output_dir = "./tmp/sdxl_model"
tar_file = "./tmp/model.tar.gz"
s3_key = "sdxl/model.tar.gz"
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
with tarfile.open(tar_file, "w:gz") as tar:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            fullpath = os.path.join(root, file)
            arcname = os.path.relpath(fullpath, start=output_dir)
            tar.add(fullpath, arcname=arcname)

print(f"Uploading {tar_file} to s3://{S3_PROJECT_BUCKET}/{s3_key}...")
s3 = boto3.client("s3")
s3.upload_file(tar_file, S3_PROJECT_BUCKET, s3_key)

print("Model Uploaded successfully to S3.")
