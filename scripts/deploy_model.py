import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

s3_model_path = "s3://job_names_classification/model.tar.gz"
role_arn = "arn:aws:iam::your-account-id:role/service-role/AmazonSageMaker-ExecutionRole"

sagemaker_session = sagemaker.Session()
huggingface_model = HuggingFaceModel(
    model_data=s3_model_path,
    role=role_arn,
    transformers_version="4.17",
    pytorch_version="1.10",
    py_version="py38",
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print(f"Model deployed at endpoint: {predictor.endpoint_name}")
