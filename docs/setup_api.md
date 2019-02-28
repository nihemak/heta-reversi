# Setup API Environment

## Create ECR repository of build api

```bash
ECR_REPO_NAME="heta-reversi-build-api"
ECR_REPO=$(aws ecr create-repository --repository-name ${ECR_REPO_NAME})
ECR_REPO_URL=$(echo ${ECR_REPO} | jq -r ".repository.repositoryUri")
cat <<EOF > ecr_policy.json
{
    "Version": "2008-10-17",
    "Statement": [
        {
            "Sid": "CodeBuildAccess",
            "Effect": "Allow",
            "Principal": {
                "Service": "codebuild.amazonaws.com"
            },
            "Action": [
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:BatchCheckLayerAvailability"
            ]
        }
    ]
}
EOF
aws ecr set-repository-policy --repository-name ${ECR_REPO_NAME} --policy-text file://ecr_policy.json
```

## Create IAM ecr build role of build api

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
ROLE_ECR_BUILD=$(aws iam create-role --role-name heta-reversi-build-api-build-ecr \
                                     --assume-role-policy-document file://Trust-Policy.json)
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AdministratorAccess \
                           --role-name heta-reversi-build-api-build-ecr
ROLE_ECR_BUILD_ARN=$(echo ${ROLE_ECR_BUILD} | jq -r ".Role.Arn")
```

## Create CodeBuild ecr of build api

```bash
cat <<EOF > Source.json
{
  "type": "CODECOMMIT",
  "location": "https://git-codecommit.ap-northeast-1.amazonaws.com/v1/repos/heta-reversi",
  "buildspec": "buildspec_api_ecr.yml"
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
    }
  ]
}
EOF
aws codebuild create-project --name heta-reversi-build-api-batch-ecr \
                               --source file://Source.json \
                               --artifacts file://Artifacts.json \
                               --environment file://Environment.json \
                               --service-role ${ROLE_ECR_BUILD_ARN}
BRANCH="master"
CODEBUILD_ID=$(aws codebuild start-build --project-name heta-reversi-build-api-batch-ecr --source-version ${BRANCH} | tr -d "\n" | jq -r '.build.id')
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

```bash
aws cloudformation validate-template --template-body file://environment.cfn.yml
aws cloudformation create-stack --stack-name heta-reversi-api --template-body file://environment.cfn.yml --capabilities CAPABILITY_IAM
```

```bash
CODEBUILD_ID=$(aws codebuild start-build --project-name heta-reversi-api --source-version master | tr -d "\n" | jq -r '.build.id')
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
