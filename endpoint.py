import pandas as pd
import sagemaker
from sagemaker.model_monitor import DataCaptureConfig

def endpoint(bucket,model_data,role,region):
    """A function that deploy a trained model as an endpoint

                        Arguments:
                        bucket -- s3 bucket
                        model_data -- s3 path for the trained model
                        role -- sagemaker role
                        region -- sagemaker region

                        """
    image_uri = sagemaker.image_uris.retrieve('image-classification',region,version='latest')
    img_classifier_model = sagemaker.Model(image_uri = image_uri,
                                 model_data = model_data,
                                 role = role)
    data_capture_config = DataCaptureConfig(
        enable_capture = True,
        sampling_percentage=100,
        destination_s3_uri=f"s3://{bucket}/data_capture"
    )
    
    deployment = img_classifier_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name='scones-unlimited-endpoint',
        data_capture_config=data_capture_config
    )
    
    endpoint = img_classifier_model.endpoint_name
    print('Endpoint created with name : {}'.format(endpoint))
    return endpoint