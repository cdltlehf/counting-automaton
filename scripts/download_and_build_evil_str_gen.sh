#!/bin/bash

URL="https://github.com/Anonymous89813/EvilStrGen.git"
mkdir -p third-party
cd third-party
git clone --depth 1 "${URL}"

cd EvilStrGen
mkdir -p build
cd build
cmake ..
cmake --build .
