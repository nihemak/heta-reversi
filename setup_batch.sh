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
