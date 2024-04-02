install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=main --cov=mylib test_*.py

format:	
	black *.py src/*.py

lint:
	pylint --disable=R,C,not-callable --ignore-patterns=test_.*?py --extension-pkg-whitelist='pydantic' --generated-members=torch.*,numpy.*,cv2.* --init-hook='import sys; sys.path.append("/workspaces/DOD-ci-cd-main/src/")' *.py src/*.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	#example deploy to aws
	#pushes container to ECR (your info will be different!)
	#aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 561744971673.dkr.ecr.us-east-1.amazonaws.com
	#docker build -t cdfastapi .
	#docker tag cdfastapi:latest 561744971673.dkr.ecr.us-east-1.amazonaws.com/cdfastapi:latest
	#docker push 561744971673.dkr.ecr.us-east-1.amazonaws.com/cdfastapi:latest

all: install lint test format deploy
