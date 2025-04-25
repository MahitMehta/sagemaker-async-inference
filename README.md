
## Autoscaling

1. `python endpoint.py autoscale`

2. Confirm Success

```
aws application-autoscaling describe-scalable-targets \
  --service-namespace sagemaker
```