import boto3
import os

ENDPOINT = os.environ['ENDPOINT']

def lambda_handler(event, context):
    runtime = boto3.Session().client('sagemaker-runtime')
    response = runtime.invoke_endpoint(EndpointName = ENDPOINT, 
            ContentType = 'text/plain',                 
            Body = event['body'])                      

    result = response['Body'].read().decode('utf-8')

    return {
            'statusCode' : 200,
            'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
            'body' : result
            }
