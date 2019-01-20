#!/bin/bash

## Create VPC
VPC=$(aws ec2 create-vpc --cidr-block 10.0.0.0/16)
VPC_ID=$(echo ${VPC} | jq -r ".Vpc.VpcId")
aws ec2 create-tags \
    --resources ${VPC_ID} \
    --tags Key=Name,Value=test-batch
