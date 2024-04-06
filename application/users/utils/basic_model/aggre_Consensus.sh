#!/bin/bash

cd ~/go/src/github.com/iot-data-sharing-project/application/users/client_$1
python3 aggregation_Consensus.py --clinum=$1 --num_of_consensus_oneround=$2