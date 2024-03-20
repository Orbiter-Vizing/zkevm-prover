#!/usr/bin/env bash
shopt -s extglob

# Install dependencies
sudo apt update && sudo apt install build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev postgresql libpqxx-dev libpqxx-doc nasm libsecp256k1-dev grpc-proto libsodium-dev libprotobuf-dev libssl-dev cmake libgrpc++-dev protobuf-compiler protobuf-compiler-grpc uuid-dev

# Download necessary files
ARCHIVE_NAME="v3.0.0-RC3-fork.6"
ARCHIVE_EXTENSION=".tgz"
ARCHIVE_URL="https://zkproverc.s3.ap-northeast-1.amazonaws.com/${ARCHIVE_NAME}${ARCHIVE_EXTENSION}"

if [ -d ${ARCHIVE_NAME} ]; then
	echo "${ARCHIVE_NAME}${ARCHIVE_EXTENSION} uncompressed"
else
	echo "${ARCHIVE_NAME}${ARCHIVE_EXTENSION} does not uncompressed, start decompressing now"
	if [ -f ${ARCHIVE_NAME}${ARCHIVE_EXTENSION} ]; then
  	echo "${ARCHIVE_NAME}${ARCHIVE_EXTENSION} already exists"
  else
  	echo "${ARCHIVE_NAME}${ARCHIVE_EXTENSION} does not exist, start downloading now"
  	wget ${ARCHIVE_URL}
  fi
	tar -xzvf ${ARCHIVE_NAME}${ARCHIVE_EXTENSION}
fi

rm -rfv config/!(*.json)
cp -R ${ARCHIVE_NAME}/config .
rm ${ARCHIVE_NAME}${ARCHIVE_EXTENSION}

# compile
# shellcheck disable=SC2164
cd src/grpc
make
cd ../../
make clean
make -j

echo "Deployment zkevm-prover complete!!!"
