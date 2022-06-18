import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler_serializerimageData(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    # TODO: fill in
    with open('/tmp/image.png', 'wb') as f:
        s3.download_fileobj(bucket,key, f)
    # We read the data from a file
    with open('/tmp/image.png','rb') as f :
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2022-06-18-01-13-56-690"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])
    runtime = boto3.client('runtime.sagemaker')
    
    inferences = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType='image/png',
        Body=image)

    # We return the data back to the Step Function
    event['body']["inferences"] = inferences['Body'].read().decode('utf-8')
    
    return {
        'statusCode': 200,
        'body': event['body']
    }

THRESHOLD = .90


def lambda_handler_filterLowConfidencePredictions(event, context):

    # Grab the inferences from the event
    probs = json.loads(event['body']['inferences'])

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any([i > THRESHOLD for i in probs])

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }