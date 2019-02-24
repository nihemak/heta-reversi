# Setup API Environment

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
