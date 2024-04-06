#!/bin/bash

echo "start train the algorithm"
cd ./model
python3 train_model.py $1
