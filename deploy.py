import time
from time import gmtime, strftime
import sagemaker

sessiong = sagemaker.Session()

model_name = ''

endpoint_config_name = 'pytorch-model' + strftime('%Y-%m-%d-%H-%M-%S', gmtime())

endpoint_config_info = session.sagemaker_client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductVariations = [{}]
