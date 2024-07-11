#!/bin/sh

# Step 0: Install necessary tools
sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y jq awscli parallel

# Step 1 Calculate SHA256 sums of configuration file
find /home/ubuntu/zkevm-prover-build/config/ -type f | parallel -j 96 sha256sum > /home/ubuntu/parallel_checksums.txt

# Step 2: Modify zkprover config file
# Get instance ID from instance metadata service
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
INSTANCE_ID=`curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id`

SECRET_AWS_REGION=${SECRET_AWS_REGION:-ap-southeast-1}

# Use AWS Secrets Manager to retrieve aggregatorClientHost and databaseURL
# Ensure that the EC2 instance has been assigned the appropriate permissions via IAM role
SECRET_ID=${ZKPROVER_SECRET_ID:-zkprover-polygon-cdk-localnet}
SECRET_STRING=$(aws secretsmanager get-secret-value --secret-id $SECRET_ID --region $SECRET_AWS_REGION --query 'SecretString' --output text)

if [ -z "$SECRET_STRING" ]; then
  echo "Failed to retrieve secrets. Exiting."
  exit 1
fi

AGGREGATOR_HOST=$(echo $SECRET_STRING | jq -r '.aggregatorClientHost')
DATABASE_URL=$(echo $SECRET_STRING | jq -r '.databaseURL')

if [ -z "$AGGREGATOR_HOST" ] || [ -z "$DATABASE_URL" ]; then
  echo "Failed to parse secrets. Exiting."
  exit 1
fi

DB_MULTI_WRITE=${DB_MULTI_WRITE:-false}
DB_READ_ONLY=${DB_READ_ONLY:-true}
STATE_MANAGER=${STATE_MANAGER:-false}

# Modify JSON configuration file
jq --arg newProverName "$INSTANCE_ID" \
   --arg newAggregatorHost "$AGGREGATOR_HOST" \
   --arg newDatabaseURL "$DATABASE_URL" \
   --argjson dbMultiWrite "$DB_MULTI_WRITE" \
   --argjson dbReadOnly "$DB_READ_ONLY" \
   --argjson stateManager "$STATE_MANAGER" \
   '.proverName = $newProverName | 
    .aggregatorClientHost = $newAggregatorHost | 
    .databaseURL = $newDatabaseURL | 
    .dbMultiWrite = $dbMultiWrite | 
    .dbReadOnly = $dbReadOnly | 
    .stateManager = $stateManager' /home/ubuntu/zkevm-prover-build/config/config_prover.json > /home/ubuntu/zkevm-prover-build/config/temp.json

# Overwrite old configuration file
mv /home/ubuntu/zkevm-prover-build/config/temp.json /home/ubuntu/zkevm-prover-build/config/config_prover.json

# Step 3: Run the service
cd /home/ubuntu/zkevm-prover-build
./tools/run_zkevm_prover.sh monitor