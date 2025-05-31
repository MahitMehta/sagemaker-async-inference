import sys
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.async_inference import AsyncInferenceConfig

from constants import MODEL_S3_KEY, S3_PROJECT_BUCKET, SAGEMAKER_ENDPOINT_NAME, SAGEMAKER_MODEL_NAME


def deploy():
    # Guide for deploying a custom inference script in SageMaker
    # https://github.com/huggingface/notebooks/blob/main/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb

    role = sagemaker.get_execution_role(use_default=True)

    # https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    model = HuggingFaceModel(
        model_data=f"s3://{S3_PROJECT_BUCKET}/{MODEL_S3_KEY}", 
        role=role,                  
        transformers_version="4.37.0",
        pytorch_version="2.1.0",     
        py_version='py310',    
        env={
            "HF_TASK": "text-to-image"
        }       
    )

    async_config = AsyncInferenceConfig(
        output_path="s3://mahitm-genai/sdxl-output/",
        max_concurrent_invocations_per_instance=1,
    )

    # Calculating cost of the endpoint
    # https://calculator.aws/#/createCalculator/SageMaker

    model.deploy(
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",  # "ml.g6.xlarge",
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
                "CustomizedMetricSpecification": {
                    "MetricName": "ApproximateBacklogSizePerInstance",
                    "Namespace": "AWS/SageMaker",
                    "Dimensions": [
                        {'Name': "EndpointName", "Value": SAGEMAKER_ENDPOINT_NAME }
                    ],
                    "Statistic": "Average",
                },
            }
        )

        # https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-autoscale.html

        step_scaling = autoscaling.put_scaling_policy(
            PolicyName="HasBacklogWithoutCapacity-ScalingPolicy",
            ServiceNamespace="sagemaker", 
            ResourceId=resource_id, 
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="StepScaling", 
            StepScalingPolicyConfiguration={
                "AdjustmentType": "ChangeInCapacity", 
                "MetricAggregationType": "Average", 
                "Cooldown": 30, 
                "StepAdjustments":
                [ 
                    {
                        "MetricIntervalLowerBound": 0,
                        "ScalingAdjustment": 1
                    }
                ]
            },    
        )

        # scale up from zero when pending requests are available
        cw_client = boto3.client("cloudwatch")

        step_scaling_policy_arn = step_scaling["PolicyARN"]
        step_scaling_policy_alarm_name = "HasBacklogWithoutCapacity-Alarm"

        cw_client.put_metric_alarm(
            AlarmName=step_scaling_policy_alarm_name,
            MetricName='HasBacklogWithoutCapacity',
            Namespace='AWS/SageMaker',
            Statistic='Average',
            EvaluationPeriods= 2,
            DatapointsToAlarm= 2,
            Threshold= 1,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            TreatMissingData='missing',
            Dimensions=[
                { "Name":'EndpointName', "Value": SAGEMAKER_ENDPOINT_NAME },
            ],
            Period= 60,
            AlarmActions=[step_scaling_policy_arn]
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
