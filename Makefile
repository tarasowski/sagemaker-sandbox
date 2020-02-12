CFN_ARTIFACTS_BUCKET := 'tarasowski-local-cs-sagemaker'
AWS_REGION :='eu-central-1'

download:
	@curl http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz --output ./input/aclImdb_v1.tar.gz
	@tar -zxf ./input/aclImdb_v1.tar.gz -C ./input/
	@rm -rf ./input/aclImdb_v1.tar.gz

preprocess:
DICT_DIR=./input CACHE_DIR=./input DATA_DIR=./input/aclImdb python3 ./src/preprocess/preprocess.py

create_bucket:
	@aws s3api create-bucket --bucket $(CFN_ARTIFACTS_BUCKET) --region $(AWS_REGION) --create-bucket-configuration LocationConstraint=$(AWS_REGION)

train-cloud:
ENV=cloud python3 jobsubmit.py

train-local:
ENV=local python3 jobsubmit.py

deploy_api:
ifdef stack_name
	@aws cloudformation package --template-file ./infrastructure/app/api.template.yaml --output-template-file ./infrastructure/app/api-output.yaml --s3-bucket $(CFN_ARTIFACTS_BUCKET) --region eu-central-1
	@aws cloudformation deploy --template-file ./infrastructure/app/api-output.yaml --stack-name $(stack_name) --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND --region $(AWS_REGION) 
else
	$(error "Please provide following arguments: stack_name=string ")
endif

deploy:
ifdef job_name
ifdef stack_name
	@aws cloudformation package --template-file ./infrastructure/app/template.yaml --output-template-file ./infrastructure/app/output.yaml --s3-bucket $(CFN_ARTIFACTS_BUCKET) --region eu-central-1
	TRAINING_JOB_NAME=$(job_name) STACK_NAME=$(stack_name) python3 ./infrastructure/app/deploy.py
else
	$(error "Please provide following arguments: job_name=string, stack_name=string ")
endif
endif




