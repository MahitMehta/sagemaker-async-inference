import json
import time
import boto3
import uuid
import urllib.parse
from botocore.exceptions import ClientError

from constants import SAGEMAKER_ENDPOINT_NAME, S3_INPUT_BUCKET, S3_INPUT_PREFIX
from extract import extract_png

def trigger_endpoint(download_output=True):
    S3_INPUT_KEY = f"{S3_INPUT_PREFIX}/prompt-{uuid.uuid4()}.json"

    input_payload = {
        "inputs": "a photo of a cat",
        "parameters": {
            "width": 1024, "guidance_scale": 10, "num_inference_steps": 30
        },
    }

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=S3_INPUT_BUCKET,
        Key=S3_INPUT_KEY,
        Body=json.dumps(input_payload),
        ContentType="application/json",
    )
    print(f"Uploaded input to: s3://{S3_INPUT_BUCKET}/{S3_INPUT_KEY}")

    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint_async(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        InputLocation=f"s3://{S3_INPUT_BUCKET}/{S3_INPUT_KEY}",
        InvocationTimeoutSeconds=120,
    )

    output_location = response["OutputLocation"]
    inference_id = response["InferenceId"]
    print(f"Inference ID: {inference_id}")
    print(f"Output will be available at: {output_location}")

    s3 = boto3.client("s3")
    parsed = urllib.parse.urlparse(output_location)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    
    if not download_output:
        print("Skipping download of output file.")
        return
    
    print("Waiting for result…")

    elapsed_time = 0
    while True:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            break
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                elapsed_time += 1
                time.sleep(1)
                print(f"Generating for {elapsed_time} seconds…")
                if elapsed_time > 30:
                    print("Timeout waiting for result.")
                    return
            else:
                raise

    obj = s3.get_object(Bucket=bucket, Key=key)
    result_bytes = obj["Body"].read()

    try:
        s3.delete_object(Bucket=bucket, Key=key)
        s3.delete_object(Bucket=bucket, Key=S3_INPUT_KEY)
        print(f"Freed temporary input from: s3://{S3_INPUT_BUCKET}/{S3_INPUT_KEY}")
        print(f"Freed temporary output from: s3://{bucket}/{key}")
    except ClientError as e:
        print(f"Error deleting output file: {e}")

    output_dir = "tmp/output.bin"
    with open(output_dir, "wb") as f:
        f.write(result_bytes)

    final_output_file_name = f"output-{uuid.uuid4()}"
    extract_png(output_dir, output_file_name=final_output_file_name)

if __name__ == "__main__":
    trigger_endpoint(download_output=True)