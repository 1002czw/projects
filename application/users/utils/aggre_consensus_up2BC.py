# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:34:47 2022

@author: czw
"""
import glob, json

import torch
import sys
sys.path.append("./basic_model")
from utils.parameters import args_parser

if __name__ == '__main__':
    # parse args
    args = args_parser()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path_sv=f'../global_Model/global_Model_Server/{args.loadpath}/cliEval.json'
    with open(path_sv, 'r') as f:
        client_sv=json.loads(f.read())
    for i in client_sv:
        client_sv[i]=[0]
    file_paths=glob.glob(f"../global_Model/global_Model_Server/{args.loadpath}/eval_out/eval_out_static_cli_*.json")
    client_checkpoints = {}
    for client_filepath in file_paths:
        with open(client_filepath, 'r') as f:
            contribution=json.loads(f.read())
        for k in contribution["Contributions"]:
            client_sv[k] += [contribution["Contributions"][k][args.reward_metrics]]
    for client_id in client_sv:
            values = client_sv[client_id]
            client_sv[client_id] = sum(values) / len(values)
    json_str = json.dumps(client_sv)
    with open(path_sv, 'w') as json_file:
        json_file.write(json_str)