from preprocess import extract_cifar_data,transform_data,get_train_test_dataframes,save_images,send_to_s3,to_metadata_file
from training import train
from endpoint import endpoint
import sagemaker

session = sagemaker.Session()

labels_to_filter = [b'bicycle',b'motorcycle']
bucket = 'project-scones-unlimited'
role = sagemaker.get_execution_role()
s3_output_location = f"s3://{bucket}/models/image_model"
region = session.boto_region_name
model_data = 's3://project-scones-unlimited/models/image_model/image-classification-2022-06-08-12-57-17-598/output/model.tar.gz'

print('\n','- -'*10,'INITIALIZING PREPROCESS','- -'*10,'\n')
#Extract data
extract_cifar_data("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")     
#Tranform extracted data
dataset_meta,dataset_test,dataset_train = transform_data()
#Get train and test data for informed labels
df_train,df_test = get_train_test_dataframes(labels_to_filter,dataset_train,dataset_test,dataset_meta)
#Extract images and save into train and test local directory
df_train.apply(lambda x : save_images(x['filenames'],x['row'],'train',dataset_train,dataset_meta),axis=1)
df_test.apply(lambda x : save_images(x['filenames'],x['row'],'test',dataset_test,dataset_meta),axis=1)
#Send images to s3
send_to_s3(bucket)
#Sending matadata for training the image classifier model
to_metadata_file(df_train.copy(), "train",bucket)
to_metadata_file(df_test.copy(), "test",bucket)

print('\n','- -'*10,'INITIALIZING TRAINING','- -'*10,'\n')
# #Training
# train(bucket,s3_output_location,role)

print('\n','- -'*10,'INITIALIZING DEPLOY','- -'*10,'\n')
endpoint_name = endpoint(bucket,model_data,role,region)
print('Endpoint created with name : {}'.format(endpoint_name))


