#!/bin/bash

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
