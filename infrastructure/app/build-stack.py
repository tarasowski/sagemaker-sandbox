import boto3
import os

TRAINING_JOB_NAME = os.environ.get('TRAINING_JOB_NAME')
STACK_NAME = os.environ.get('STACK_NAME')

dir_path = os.path.dirname(os.path.realpath(__file__))

smc = boto3.client('sagemaker')

job = smc.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)
roleArn = job['RoleArn']
modelDataUrl = job['ModelArtifacts']['S3ModelArtifacts']
trainingImage = job['AlgorithmSpecification']['TrainingImage']

cfn = boto3.client('cloudformation')

with open(dir_path + '/template.yaml', 'r') as f:
    stack = cfn.create_stack(StackName=STACK_NAME,
            TemplateBody = f.read(),
            Parameters=[
                    {'ParameterKey':'ModelName', 'ParameterValue': TRAINING_JOB_NAME},
                    {'ParameterKey':'TrainingImage', 'ParameterValue': trainingImage},
                    {'ParameterKey':'ModelDataUrl', 'ParameterValue': modelDataUrl},
                    {'ParameterKey':'RoleArn', 'ParameterValue': roleArn}])
