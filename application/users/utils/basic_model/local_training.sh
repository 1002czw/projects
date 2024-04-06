#!/bin/bash

echo "yes"
cd ~/go/src/github.com/iot-data-sharing-project/application/users/client_$1
python3 local_Training.py --dp_mechanism=$2 --dp_clip=$3 --dp_epsilon=$4