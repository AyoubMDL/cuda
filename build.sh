#!/bin/bash

# Check if a day is specified
if [ -z "$1" ]; then
    echo "Usage: ./build.sh <day>"
    exit 1
fi

DAY=$1

# Create build directory
mkdir -p build

cd build

# Run CMake with the specified day
cmake -DDAY=${DAY} ..

# Build the project
make

# Run the executable (example: ./build.sh day001)
./${DAY}/main