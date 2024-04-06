#!/bin/bash

echo "start to initialize client local model"

cd ~/go/src/github.com/iot-data-sharing-project/application/users/client_$1
python3 downModel_Initialization.py --clinum=$1 --loadpath=$2

cd ~/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server/ShapleyValue/server
cp server_model.pt test_data.pt ~/go/src/github.com/iot-data-sharing-project/application/users/client_$1/ShapleyValue/train_model
