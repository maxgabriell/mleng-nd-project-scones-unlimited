import os

#os.system('pip install jsonlines')

import random
import boto3
import json
import jsonlines
import matplotlib.pyplot as plt
from sagemaker.s3 import S3Downloader
import argparse
import time

def generate_test_case(bucket):
    # Setup s3 in boto3
    s3 = boto3.resource('s3')
    
    # Randomly pick from sfn or test folders in our bucket
    objects = s3.Bucket(bucket).objects.filter(Prefix = "test")
    
    # Grab any random object key from that folder!
    obj = random.choice([x.key for x in list(objects)])
    
    return json.dumps({
        "image_data": "",
        "s3_bucket": bucket,
        "s3_key": obj
    })


def simple_getter(obj):
    '''Define how we'll get our data'''
    inferences = obj["captureData"]["endpointOutput"]["data"]
    timestamp = obj["eventMetadata"]["inferenceTime"]
    return json.loads(inferences), timestamp

bucket = 'project-scones-unlimited'
stateMachineArn = 'arn:aws:states:us-east-1:820235860091:stateMachine:scones-unlimited-workflow'
data_path = 's3://project-scones-unlimited/data_capture/image-classification-2022-06-12-17-39-54-093/AllTraffic/2022/06/12/19/'
#Getting time for simulation execution
parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--t", type=int, default=1)
args = parser.parse_args()
time_execution_min = args.t
step_func_cli = boto3.client('stepfunctions')

t_end = time.time() + 60 * time_execution_min
t_start = time.time()
print('\n','- -'*5,f'SIMULATION START AT {t_start}','- - '*5,'\n')
while time.time() < t_end:
    print('\n','- -'*2,f'SOLICITATION RECEIVED AFTER {time.time() - t_start}s','- - '*2,'\n')
    name = str(hash(random.random()))
    state_machine_exec = step_func_cli.start_execution(
        stateMachineArn=stateMachineArn, # You can find this through the Console or through the 'response' object. 
        name=name, # Execution names need to be unique within state machines. 
        input=generate_test_case(bucket) # Input needs to be at least empty brackets. 
    )
    print('FILE SUBMITED TO STATE MACHINE : {}'.format(state_machine_exec))
    seconds_to_sleep = abs(round(random.gauss(15, 5)))
    time.sleep(seconds_to_sleep)

# S3Downloader.download(data_path, "captured_data")

# # List the file names we downloaded
# file_handles = os.listdir("./captured_data")

# # Dump all the data into an array
# json_data = []
# for jsonl in file_handles:
#     with jsonlines.open(f"./captured_data/{jsonl}") as f:
#         json_data.append(f.read())
        
# # Populate the data for the x and y axis
# x = []
# y = []
# for obj in json_data:
#     inference, timestamp = simple_getter(obj)
    
#     y.append(max(inference))
#     x.append(timestamp)

# # Plot the data
# plt.scatter(x, y, c=['r' if k<.94 else 'b' for k in y ])
# plt.axhline(y=0.9, color='g', linestyle='--')
# plt.ylim(bottom=.88)

# # Add labels
# plt.ylabel("Confidence")
# plt.suptitle("Observed Recent Inferences", size=14)
# plt.title("Pictured with confidence threshold for production use", size=10)

# # Give it some pizzaz!
# plt.style.use("Solarize_Light2")
# plt.gcf().autofmt_xdate()
# plt.show()