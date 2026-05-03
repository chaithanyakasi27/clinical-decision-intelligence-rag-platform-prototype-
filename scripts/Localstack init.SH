#!/bin/bash
# scripts/localstack_init.sh
# Runs automatically when LocalStack starts — creates mock AWS resources
 
echo "→ Initializing LocalStack AWS resources..."
 
# S3 buckets
awslocal s3 mb s3://cdip-raw-clinical-docs-dev
awslocal s3 mb s3://cdip-processed-docs-dev
echo "✓ S3 buckets created"
 
# DynamoDB table for document metadata
awslocal dynamodb create-table \
  --table-name cdip-metadata-dev \
  --attribute-definitions \
    AttributeName=patient_id,AttributeType=S \
    AttributeName=document_id,AttributeType=S \
  --key-schema \
    AttributeName=patient_id,KeyType=HASH \
    AttributeName=document_id,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
echo "✓ DynamoDB table created"
 
echo "✓ LocalStack init complete"
 