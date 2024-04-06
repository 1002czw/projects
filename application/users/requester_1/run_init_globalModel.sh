#!/bin/bash

echo "start to initialize Global Model"
cd ~/go/src/github.com/iot-data-sharing-project/application/users/requester_1/
python3 init_globalModel.py
rm -r ~/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server
cp -r ./global_Model_Server ~/go/src/github.com/iot-data-sharing-project/application/users/global_Model
