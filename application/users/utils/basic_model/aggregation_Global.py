# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:10:16 2022

@author: czw
"""
import glob
from utils.parameters import args_parser
from models.Fed import FedAvg
from utils.save_file import createDirectory, deleteAllModels, saveCheckpoint, print_parameters, loadCheckpoint
import torch
from models.Nets import SmallMLP_MNIST, MediumMLP_MNIST, LargeMLP_MNIST, SmallMLP_EMNIST, MediumMLP_EMNIST, LargeMLP_EMNIST
from torch import nn
import json

def aggregation_Global(server):

    # Load client model state_dicts (simulating server sideloading client model parameters)
    client_filepaths = glob.glob(f"../global_Model/global_Model_Server/{args.loadpath}/client/client*.pt")
    NUM_OF_CLIENTS = args.num_normal_clients + args.num_freerider_clients + args.num_adversarial_clients
    newclient_checkpoints = {}
    for client_filepath in client_filepaths:
        client_checkpoint = loadCheckpoint(client_filepath)
        newclient_checkpoints[client_checkpoint['name']] = client_checkpoint
        #newclient_checkpoints += [client_checkpoint]
    client_names =[client_id for client_id in newclient_checkpoints]
    #client_names = []
    #for i in range(NUM_OF_CLIENTS):
        #client = f'client_{i+1}'
        #client_names.append(client)

    #/home/czw/go/src/github.com/iot-data-sharing-project/application/users/global_Model/global_Model_Server/ShapleyValue/cliEval.json
    #./client_fraction.json
    with open('./client_fraction.json', 'r') as f:
        fraction_dict=json.loads(f.read())
    # Get Federated Average of clients' parameters

    model_state_dicts = [newclient_checkpoints[client_id] ['model_state_dict'] for client_id in client_names]
    client_weights=[fraction_dict[client_id] for client_id in client_names]

    fed_model_state_dict = FedAvg(model_state_dicts,client_weights)

    # Instantiate server model using FedAvg
    newfed_model = FederatedModel().to(device)
    newfed_model.load_state_dict(fed_model_state_dict)
    newfed_model.eval()#
    save_server_file=f'../global_Model/global_Model_Server/{args.loadpath}/server/server_model.pt'
    saveCheckpoint(
        server['name'],
        newfed_model.state_dict(),
        server['optimizer_state_dict'],
        save_server_file,
    )

if __name__ == '__main__':
    # parse args
    args = args_parser()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FederatedModel = None

    if (args.model_size == 'SMALL') & (args.dataset_type == 'MNIST'):
        FederatedModel = SmallMLP_MNIST
    elif (args.model_size == 'MEDIUM') & (args.dataset_type == 'MNIST'):
        FederatedModel = MediumMLP_MNIST
    elif (args.model_size == 'LARGE') & (args.dataset_type == 'MNIST'):
        FederatedModel = LargeMLP_MNIST
    elif (args.model_size == 'SAMLL') & (args.dataset_type == 'EMNIST'):
        FederatedModel = SmallMLP_EMNIST
    elif (args.model_size == 'MEDIUM') & (args.dataset_type == 'EMNIST'):
        FederatedModel = MediumMLP_MNIST
    elif (args.model_size == 'LARGE') & (args.dataset_type == 'EMNIST'):
        FederatedModel = LargeMLP_EMNIST

    # Define network training functions and hyper-parameters
    # Training hyper-parameters and functions for the Federated modeel
    FederatedLossFunc = nn.CrossEntropyLoss
    FederatedOptimizer = torch.optim.SGD
    FederatedLearnRate = args.learning_rate
    FederatedMomentum = args.momentum
    FederatedWeightDecay = args.weight_decay

    server = torch.load(f'./{args.loadpath}/train_model/server_model.pt')
    aggregation_Global(server)
