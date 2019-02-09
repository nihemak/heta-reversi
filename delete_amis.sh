#!/bin/bash

aws ec2 describe-images \
    --owner self \
    --filters "Name=tag:USE,Values=heta-reversi" | jq -r ".Images[].ImageId" | while read -r AMI_ID
do
  aws ec2 deregister-image --image-id ${AMI_ID}
done
