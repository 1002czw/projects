#!/bin/bash
cd ~/go/src/github.com/iot-data-sharing-project/application/users/client_$1/$2/train_model
cp client_$1.pt ~/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server/$2/client