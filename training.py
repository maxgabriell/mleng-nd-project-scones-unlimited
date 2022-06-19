import pandas as pd
import sagemaker
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput

def train(bucket,s3_output,role):
    """A function that trains a image classifier model and saves it in s3

                    Arguments:
                    bucket -- s3 bucket
                    s3_output -- s3 path for the trained model
                    role -- sagemaker role

                    """
    # Use the image_uris function to retrieve the latest 'image-classification' image 
    algo_image = sagemaker.image_uris.retrieve('image-classification',region,version='latest')

    #Create Estimator
    img_classifier_model=sagemaker.estimator.Estimator(
        image_uri= algo_image, #The container where the code image is stored with the model code, OS etc.. 
        role= role, #The role you`re uysing that garantees permissions 
        instance_count= 1, #The number of instances (machines) used
        instance_type= 'ml.p2.xlarge', #The type of the instance used
        output_path= s3_output, #where to save the output (model artifact)
        sagemaker_session=sagamaker.Session()
    )
    #Setting hyperparameters
    img_classifier_model.set_hyperparameters(
        image_shape='3,32,32',
        num_classes=2, 
        num_training_samples=df_train.shape[0]
    )
    #Model inputs and training
    model_inputs = {
            "train": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{bucket}/train/",
                content_type="application/x-image"
            ),
            "validation": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{bucket}/test/",
                content_type="application/x-image"
            ),
            "train_lst": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{bucket}/train.lst",
                content_type="application/x-image"
            ),
            "validation_lst": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{bucket}/test.lst",
                content_type="application/x-image"
            )
    }
    img_classifier_model.fit(model_inputs)