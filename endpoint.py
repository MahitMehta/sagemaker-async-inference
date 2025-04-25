import sys
import sagemaker
from sagemaker.model import Model
from sagemaker.async_inference import AsyncInferenceConfig
from typing import Dict
from sagemaker.workflow.parameters import PipelineVariable

from constants import MODEL_ID, SAGEMAKER_ENDPOINT_NAME, SAGEMAKER_MODEL_NAME


def deploy():
    role = sagemaker.get_execution_role(use_default=True)

    # https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    image_uri = "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04"

    env: Dict[str, str | PipelineVariable] | None = {
        "HF_TASK": "text-to-image",
        "HF_MODEL_ID": MODEL_ID
    }

    model = Model(
        image_uri=image_uri,
        role=role,
        env=env,
        sagemaker_session=sagemaker.Session(),
        # source_dir="code",
        # dependencies=["code/requirements.txt"],
        name=SAGEMAKER_MODEL_NAME,
        # entry_point="inference.py", 
    )

    # TODO: Using a custom inference script is not supported when using HF_TASK, you have gzip the code dir and upload it to S3
    # https://github.com/huggingface/notebooks/blob/main/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb

    async_config = AsyncInferenceConfig(
        output_path="s3://mahitm-genai/sdxl-output/",
        max_concurrent_invocations_per_instance=1,
    )

    # Calculating cost of the endpoint
    # https://calculator.aws/#/createCalculator/SageMaker
    
    model.deploy(
        initial_instance_count=1,
        instance_type="ml.g6.xlarge",
        async_inference_config=async_config,
        endpoint_name=SAGEMAKER_ENDPOINT_NAME,
    )


def delete():
    try:
        sagemaker.Session().delete_endpoint(endpoint_name=SAGEMAKER_ENDPOINT_NAME)
    except Exception as e:
        print(f"Error deleting endpoint: {e}")

    try:
        sagemaker.Session().delete_endpoint_config(
            endpoint_config_name=SAGEMAKER_ENDPOINT_NAME
        )
    except Exception as e:
        print(f"Error deleting endpoint config: {e}")

    try:
        sagemaker.Session().delete_model(model_name=SAGEMAKER_MODEL_NAME)
    except Exception as e:
        print(f"Error deleting model: {e}")


def autoscale():
    import boto3

    autoscaling = boto3.client("application-autoscaling")
    resource_id = f"endpoint/{SAGEMAKER_ENDPOINT_NAME}/variant/AllTraffic"

    try:
        autoscaling.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=0,
            MaxCapacity=1,
        )

        autoscaling.put_scaling_policy(
            PolicyName="AsyncScaleToZeroPolicy",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": 1.0,
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                },
                "ScaleInCooldown": 60,
                "ScaleOutCooldown": 60,
                "DisableScaleIn": False,
            },
        )
    except Exception as e:
        print(f"Error configuring autoscaling: {e}")


if __name__ == "__main__":
    if sys.argv[1] == "deploy":
        deploy()
        autoscale()
    elif sys.argv[1] == "delete":
        delete()
    elif sys.argv[1] == "redeploy":
        delete()
        deploy()
        autoscale()
    elif sys.argv[1] == "autoscale":
        autoscale()
    else:
        print("Invalid argument. Use 'deploy' or 'delete'.")
        sys.exit(1)
