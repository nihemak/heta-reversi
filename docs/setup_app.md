# Setup APP Environment

## Create ECR repository of app

```bash
ECR_REPO_NAME="heta-reversi-app"
ECR_REPO=$(aws ecr create-repository --repository-name ${ECR_REPO_NAME})
ECR_REPO_URL=$(echo ${ECR_REPO} | jq -r ".repository.repositoryUri")
```

## Create IAM ecr build role of app

```bash
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
ROLE_ECR_BUILD=$(aws iam create-role --role-name heta-reversi-app-ecr \
                                     --assume-role-policy-document file://Trust-Policy.json)
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AdministratorAccess \
                           --role-name heta-reversi-app-ecr
ROLE_ECR_BUILD_ARN=$(echo ${ROLE_ECR_BUILD} | jq -r ".Role.Arn")
```

## Create CodeBuild ecr of app

```bash
cat <<EOF > Source.json
{
  "type": "CODECOMMIT",
  "location": "https://git-codecommit.ap-northeast-1.amazonaws.com/v1/repos/heta-reversi",
  "buildspec": "buildspec_app.yml"
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
      "value": "heta_reversi_build_api",
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
    },
    {
      "name": "MODEL_BUCKET",
      "value": "test-batch-bucket-name",
      "type": "PLAINTEXT"
    }
  ]
}
EOF
aws codebuild create-project --name heta-reversi-app-ecr \
                               --source file://Source.json \
                               --artifacts file://Artifacts.json \
                               --environment file://Environment.json \
                               --service-role ${ROLE_ECR_BUILD_ARN}
BRANCH="master"
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
aws ecs create-cluster --cluster-name heta-reversi

IMAGE_REPO_URI=$(aws ecr describe-repositories --repository-name heta-reversi-app | jq -r '.repositories[].repositoryUri')

cat <<EOF > Trust-Policy.json
{
    "Version": "2008-10-17",
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
ROLE_ECS_TASK_EXEC=$(aws iam create-role --role-name heta-reversi-ecsTaskExecutionRole \
                                     --assume-role-policy-document file://Trust-Policy.json)
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
                           --role-name heta-reversi-ecsTaskExecutionRole
ROLE_ECS_TASK_EXEC_ARN=$(echo ${ROLE_ECS_TASK_EXEC} | jq -r ".Role.Arn")

aws logs create-log-group --log-group-name /ecs/heta-reversi

cat <<EOF > task_definition.json
{
    "family": "heta-reversi",
    "containerDefinitions": [
        {
            "name": "heta-reversi",
            "image": "${IMAGE_REPO_URI}:latest",
            "cpu": 0,
            "portMappings": [{
                "containerPort": 5000,
                "hostPort": 5000,
		            "protocol": "tcp"
            }],
            "essential": true,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-stream-prefix": "ecs",
                    "awslogs-region": "ap-northeast-1",
                    "awslogs-group": "/ecs/heta-reversi"
                }
            }
        }
    ],
    "cpu": "256",
    "memory": "512",
    "networkMode": "awsvpc",
    "executionRoleArn": "${ROLE_ECS_TASK_EXEC_ARN}",
    "requiresCompatibilities": [
        "FARGATE"
    ]
}
EOF
aws ecs register-task-definition --cli-input-json file://task_definition.json

VPC_ID=$(aws ec2 describe-vpcs --filters Name=tag:Name,Values=test-batch | jq -r '.Vpcs[].VpcId')
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name app-sg --description "app-sg" --vpc-id ${VPC_ID} | jq -r '.GroupId')
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 5000 --cidr 0.0.0.0/0

SUBNET_ID=$(aws ec2 describe-subnets --filters Name=tag:Name,Values=test-batch | jq -r '.Subnets[].SubnetId')

cat <<EOF > service_define.json
{
    "cluster": "heta-reversi",
    "serviceName": "heta-reversi_app",
    "taskDefinition": "heta-reversi",
    "desiredCount": 1,
    "launchType": "FARGATE",
    "platformVersion": "LATEST",
    "deploymentConfiguration": {
        "minimumHealthyPercent": 100,
        "maximumPercent": 200
    },
    "networkConfiguration": {
        "awsvpcConfiguration": {
            "securityGroups": [
                "${SECURITY_GROUP_ID}"
            ],
            "subnets": [
                "${SUBNET_ID}"
            ],
            "assignPublicIp": "ENABLED"
        }
    },
    "schedulingStrategy": "REPLICA"
}
EOF
aws ecs create-service --service-name heta-reversi_app --cli-input-json file://service_define.json
```
