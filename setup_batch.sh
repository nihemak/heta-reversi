#!/bin/bash

AWS_IDENTITY=$(aws sts get-caller-identity)
AWS_ACCOUNT_ID=$(echo ${AWS_IDENTITY} | jq -r ".Account")

## Create CodeCommit
aws codecommit create-repository --repository-name heta-reversi
git clone --mirror https://github.com/nihemak/heta-reversi.git
cd heta-reversi.git
git push ssh://git-codecommit.ap-northeast-1.amazonaws.com/v1/repos/heta-reversi --all
cd ..

## Create VPC
VPC=$(aws ec2 create-vpc --cidr-block 10.0.0.0/16)
VPC_ID=$(echo ${VPC} | jq -r ".Vpc.VpcId")
aws ec2 create-tags \
    --resources ${VPC_ID} \
    --tags Key=Name,Value=test-batch

## Create Internet Gateway
INTERNET_GATEWAY=$(aws ec2 create-internet-gateway)
INTERNET_GATEWAY_ID=$( \
    echo ${INTERNET_GATEWAY} | jq -r ".InternetGateway.InternetGatewayId")
aws ec2 create-tags \
    --resources ${INTERNET_GATEWAY_ID} \
    --tags Key=Name,Value=test-batch
aws ec2 attach-internet-gateway \
    --internet-gateway-id ${INTERNET_GATEWAY_ID} \
    --vpc-id ${VPC_ID}

## Create Subnet
SUBNET=$( \
    aws ec2 create-subnet \
        --vpc-id ${VPC_ID} \
        --cidr-block 10.0.0.0/24 \
        --availability-zone ap-northeast-1a)
SUBNET_ID=$(echo ${SUBNET} | jq -r ".Subnet.SubnetId")
aws ec2 create-tags \
    --resources ${SUBNET_ID} \
    --tags Key=Name,Value=test-batch
aws ec2 modify-subnet-attribute --subnet-id ${SUBNET_ID} --map-public-ip-on-launch

## Create Route Table
ROUTE_TABLE_ID=$( \
    aws ec2 describe-route-tables \
        --filters Name=vpc-id,Values=${VPC_ID} \
             | jq -r ".RouteTables[].RouteTableId")
aws ec2 create-tags \
    --resources ${ROUTE_TABLE_ID} \
    --tags Key=Name,Value=test-batch
aws ec2 create-route \
    --route-table-id ${ROUTE_TABLE_ID} \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id ${INTERNET_GATEWAY_ID}
aws ec2 associate-route-table \
    --route-table-id ${ROUTE_TABLE_ID} \
    --subnet-id ${SUBNET_ID}

## Security Group
DEFAULT_SECURITY_GROUP_ID=$( \
    aws ec2 describe-security-groups \
        --filters Name=group-name,Values=default Name=vpc-id,Values=${VPC_ID} \
            | jq -r '.SecurityGroups[].GroupId')

## Create S3
aws s3 mb s3://test-batch-bucket-name --region ap-northeast-1

## Create ECR repository
ECR_REPO_NAME="test-batch"
ECR_REPO=$(aws ecr create-repository --repository-name ${ECR_REPO_NAME})
ECR_REPO_URL=$(echo ${ECR_REPO} | jq -r ".repository.repositoryUri")

## Create IAM ecr build role
cat <<EOF > Trust-Policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "codebuild.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
ROLE_ECR_BUILD=$(aws iam create-role --role-name test-batch-build-ecr \
                                     --assume-role-policy-document file://Trust-Policy.json)
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AdministratorAccess \
                           --role-name test-batch-build-ecr
ROLE_ECR_BUILD_ARN=$(echo ${ROLE_ECR_BUILD} | jq -r ".Role.Arn")

## Create CodeBuild ecr
cat <<EOF > Source.json
{
  "type": "CODECOMMIT",
  "location": "https://git-codecommit.ap-northeast-1.amazonaws.com/v1/repos/heta-reversi",
  "buildspec": "buildspec_ecr.yml"
}
EOF
cat <<EOF > Artifacts.json
{
  "type": "NO_ARTIFACTS"
}
EOF
cat <<EOF > Environment.json
{
  "type": "LINUX_CONTAINER",
  "image": "aws/codebuild/docker:18.09.0",
  "computeType": "BUILD_GENERAL1_SMALL",
  "environmentVariables": [
    {
      "name": "REGION",
      "value": "ap-northeast-1",
      "type": "PLAINTEXT"
    },
    {
      "name": "IMAGE_REPO_NAME",
      "value": "example",
      "type": "PLAINTEXT"
    },
    {
      "name": "IMAGE_TAG",
      "value": "latest",
      "type": "PLAINTEXT"
    },
    {
      "name": "ECR_REPO_URL",
      "value": "${ECR_REPO_URL}",
      "type": "PLAINTEXT"
    }
  ]
}
EOF
aws codebuild create-project --name test-batch-ecr \
                               --source file://Source.json \
                               --artifacts file://Artifacts.json \
                               --environment file://Environment.json \
                               --service-role ${ROLE_ECR_BUILD_ARN}
CODEBUILD_ID=$(aws codebuild start-build --project-name test-batch-ecr --source-version master | tr -d "\n" | jq -r '.build.id')
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

## Create Instance role
cat <<EOF > Trust-Policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
aws iam create-role --role-name test-batch-instance \
                    --assume-role-policy-document file://Trust-Policy.json
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role \
                           --role-name test-batch-instance
INSTANCE_ROLE=$(aws iam create-instance-profile --instance-profile-name test-batch-instance)
INSTANCE_ROLE_ARN=$(echo ${INSTANCE_ROLE} | jq -r ".InstanceProfile.Arn")
aws iam add-role-to-instance-profile --role-name test-batch-instance --instance-profile-name test-batch-instance

## Create IAM ecr service role
cat <<EOF > Trust-Policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "batch.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
ROLE_SERVICE=$(aws iam create-role --role-name test-batch-service \
                                   --assume-role-policy-document file://Trust-Policy.json)
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole \
                           --role-name test-batch-service
ROLE_SERVICE_ARN=$(echo ${ROLE_SERVICE} |jq -r ".Role.Arn")

## Create Batch compute environment
cat << EOF > compute-environment.spec.json
{
    "computeEnvironmentName": "test-compute-environment",
    "type": "MANAGED",
    "state": "ENABLED",
    "computeResources": {
        "type": "EC2",
        "minvCpus": 0,
        "maxvCpus": 4,
        "desiredvCpus": 0,
        "instanceTypes": ["optimal"],
        "subnets": ["${SUBNET_ID}"],
        "securityGroupIds": ["${DEFAULT_SECURITY_GROUP_ID}"],
        "instanceRole": "${INSTANCE_ROLE_ARN}"
    },
    "serviceRole": "${ROLE_SERVICE_ARN}"
}
EOF
COMPUTE_ENV=$(aws batch create-compute-environment --cli-input-json file://compute-environment.spec.json)
COMPUTE_ENV_ARN=$(echo ${COMPUTE_ENV} | jq -r '.computeEnvironmentArn')

## Create Batch job queue
JOB_QUEUE=$(aws batch create-job-queue \
  --job-queue-name test-job-queue \
  --priority 1 \
  --compute-environment-order order=1,computeEnvironment=${COMPUTE_ENV_ARN})
JOB_QUEUE_ARN=$(echo ${JOB_QUEUE} | jq -r '.jobQueueArn')

## Create IAM job role
cat <<EOF > Trust-Policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ecs-tasks.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
ROLE_JOB=$(aws iam create-role --role-name test-batch-job \
                               --assume-role-policy-document file://Trust-Policy.json)
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
                           --role-name test-batch-job
ROLE_JOB_ARN=$(echo ${ROLE_JOB} | jq -r ".Role.Arn")

## Create job definition
cat << EOF > job-definition.spec.json
{
  "image": "${ECR_REPO_URL}",
  "vcpus": 4,
  "memory": 2000,
  "jobRoleArn": "${ROLE_JOB_ARN}"
}
EOF
JOB_DEF=$(aws batch register-job-definition \
  --job-definition-name test-job-definition \
  --type container \
  --container-properties file://job-definition.spec.json)
JOB_DEF_ARN=$(echo ${JOB_DEF} | jq -r '.jobDefinitionArn')

## Submit job
JOB=$(aws batch submit-job \
    --job-name "test-job" \
    --job-queue "${JOB_QUEUE_ARN}" \
    --job-definition "${JOB_DEF_ARN}")
JOB_ID=$(echo ${JOB} | jq -r ".jobId")

## Show job status
aws batch describe-jobs --jobs ${JOB_ID} | jq -r ".jobs[].status"

## Get model file
aws s3 ls test-batch-bucket-name/data/
aws s3 sync s3://test-batch-bucket-name/data .
