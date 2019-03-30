# Setup Batch Environment

## Define Basic Vars

```bash
BRANCH="master"
```

## Create CodeCommit

```bash
git clone --mirror https://github.com/nihemak/heta-reversi.git
cd heta-reversi.git
aws cloudformation validate-template \
    --template-body file://infra/codecommit.cfn.yml
aws cloudformation create-stack \
    --stack-name heta-reversi-codecommit \
    --template-body file://infra/codecommit.cfn.yml
git push ssh://git-codecommit.ap-northeast-1.amazonaws.com/v1/repos/heta-reversi --all
```

## Create VPC

```bash
aws cloudformation validate-template \
    --template-body file://infra/vpc.cfn.yml
aws cloudformation create-stack \
    --stack-name heta-reversi-vpc \
    --template-body file://infra/vpc.cfn.yml
```

## Create model create ami

```bash
aws cloudformation validate-template \
    --template-body file://infra/model_create_ami.cfn.yml
aws cloudformation create-stack \
    --stack-name heta-reversi-model-create-ami \
    --capabilities CAPABILITY_NAMED_IAM \
    --template-body file://infra/model_create_ami.cfn.yml
```

```bash
CODEBUILD_ID=$(aws codebuild start-build --project-name heta-reversi-model-create-ami --source-version ${BRANCH} | tr -d "\n" | jq -r '.build.id')
echo "started.. id is ${CODEBUILD_ID}"
while true
do
  sleep 10s
  STATUS=$(aws codebuild batch-get-builds --ids "${CODEBUILD_ID}" | tr -d "\n" | jq -r '.builds[].buildStatus')
  echo "..status is ${STATUS}."
  if [ "${STATUS}" != "IN_PROGRESS" ]; then
    if [ "${STATUS}" != "SUCCEEDED" ]; then
      echo "faild."
    fi
    echo "done."
    break
  fi
done
AMI_IDS=$(aws ec2 describe-images \
    --owner self \
    --filters "Name=tag:USE,Values=heta-reversi" | jq -r ".Images[].ImageId")
AMI_ID="${AMI_IDS[0]}"
```

## Create model create

```bash
aws cloudformation validate-template \
    --template-body file://infra/model_create.cfn.yml
aws cloudformation create-stack \
    --stack-name heta-reversi-model-create \
    --capabilities CAPABILITY_NAMED_IAM \
    --template-body file://infra/model_create.cfn.yml \
    --parameters \
      ParameterKey=AMI,ParameterValue=${AMI_ID}
```

```bash
CODEBUILD_ID=$(aws codebuild start-build --project-name heta-reversi-model-create-ecr --source-version ${BRANCH} | tr -d "\n" | jq -r '.build.id')
echo "started.. id is ${CODEBUILD_ID}"
while true
do
  sleep 10s
  STATUS=$(aws codebuild batch-get-builds --ids "${CODEBUILD_ID}" | tr -d "\n" | jq -r '.builds[].buildStatus')
  echo "..status is ${STATUS}."
  if [ "${STATUS}" != "IN_PROGRESS" ]; then
    if [ "${STATUS}" != "SUCCEEDED" ]; then
      echo "faild."
    fi
    echo "done."
    break
  fi
done

JOB_QUEUE_ARN=$(aws cloudformation describe-stacks --stack-name heta-reversi-model-create | jq -r '.Stacks[].Outputs[] | select(.OutputKey == "ModelCreateJobQueue") | .OutputValue')
JOB_DEF_ARN=$(aws cloudformation describe-stacks --stack-name heta-reversi-model-create | jq -r '.Stacks[].Outputs[] | select(.OutputKey == "ModelCreateJobDefinition") | .OutputValue')
```

## Submit job

```bash
JOB=$(aws batch submit-job \
    --job-name "test-job" \
    --job-queue "${JOB_QUEUE_ARN}" \
    --job-definition "${JOB_DEF_ARN}")
JOB_ID=$(echo ${JOB} | jq -r ".jobId")
```

## Show job status

```bash
aws batch describe-jobs --jobs ${JOB_ID} | jq -r ".jobs[].status"
```

## Get model file

```bash
aws s3 ls test-batch-bucket-name/data/
aws s3 sync s3://test-batch-bucket-name/data .
```

# Setup APP Environment

## Create CodeBuild ecr of app

```bash
aws cloudformation validate-template \
    --template-body file://infra/app_ecr.cfn.yml
aws cloudformation create-stack \
    --stack-name heta-reversi-app-ecr \
    --capabilities CAPABILITY_NAMED_IAM \
    --template-body file://infra/app_ecr.cfn.yml
```

```bash
CODEBUILD_ID=$(aws codebuild start-build --project-name heta-reversi-app-ecr --source-version ${BRANCH} | tr -d "\n" | jq -r '.build.id')
echo "started.. id is ${CODEBUILD_ID}"
while true
do
  sleep 10s
  STATUS=$(aws codebuild batch-get-builds --ids "${CODEBUILD_ID}" | tr -d "\n" | jq -r '.builds[].buildStatus')
  echo "..status is ${STATUS}."
  if [ "${STATUS}" != "IN_PROGRESS" ]; then
    if [ "${STATUS}" != "SUCCEEDED" ]; then
      echo "faild."
    fi
    echo "done."
    break
  fi
done
```

## Create ECS

```bash
aws cloudformation validate-template \
    --template-body file://infra/app.cfn.yml
aws cloudformation create-stack \
    --stack-name heta-reversi-app \
    --capabilities CAPABILITY_NAMED_IAM \
    --template-body file://infra/app.cfn.yml
```
