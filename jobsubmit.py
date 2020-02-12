from sagemaker.pytorch import PyTorch
import sagemaker
import os
import time
from time import gmtime, strftime

# Initialize SDK

role = 'SageMakerRole' 
input_data = os.environ.get('INPUT_DATA')
ENV = os.environ.get('ENV')

if ENV == 'local':
    code_prefix = "pytorch-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    session = sagemaker.Session()
    bucket = session.default_bucket()
    prefix = 'sagemaker/sentiment_rnn'
    role = 'SageMakerRole'
    input_data = session.upload_data(path='./input', bucket=bucket, key_prefix=prefix)
    estimator = PyTorch(entry_point='/train.py',
            source_dir='./src/train',
            role=role,
            framework_version='0.4.0',
            train_instance_type='local',
            train_instance_count=1,
            hyperparameters = {
                    'epochs': 5,
                    'hidden_dim': 200
                    })

    estimator.fit({'training': input_data})

if ENV == 'cloud':
    session = sagemaker.Session()
    bucket = session.default_bucket()
    prefix = 'sagemaker/sentiment_rnn'
    role = 'SageMakerRole'
    input_data = session.upload_data(path='./input', bucket=bucket, key_prefix=prefix)
    estimator = PyTorch(entry_point='train.py',
            source_dir='./src/train_predict/',
            role=role,
            framework_version='0.4.0',
            train_instance_count=1,
            train_instance_type='ml.p2.xlarge',
            hyperparameters = {
                    'epochs': 5,
                    'hidden_dim': 200
                    })

    estimator.fit({'training': input_data})

