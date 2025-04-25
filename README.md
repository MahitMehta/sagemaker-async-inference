## Overview

Simple utilities to deploy Hugging Face models to AWS SageMaker Async. Endpoint for inference.
Currently the code is configured for deploying SDXL.

## General Deployment

1. Configure constants in `constants.py`
2. `python endpoint.py deploy`

## Enable Autoscaling to Zero

1. `python endpoint.py autoscale`

2. Confirm Success

```
aws application-autoscaling describe-scalable-targets \
  --service-namespace sagemaker
```

3. See Recent Autoscaling Events

```
aws application-autoscaling describe-scaling-activities \
  --service-namespace sagemaker
```