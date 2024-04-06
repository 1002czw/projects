# Load libraries
import math, random, copy, os, glob, time
from itertools import chain, combinations, permutations
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import datasets, transforms as T

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from utils.parameters import args_parser
from utils.data_process import prepareIID, prepareNIID1, prepareNIID2, prepareNIID12
from models.Nets import SmallMLP_MNIST, MediumMLP_MNIST, LargeMLP_MNIST, SmallMLP_EMNIST, MediumMLP_EMNIST, LargeMLP_EMNIST
from utils.save_file import createDirectory, deleteAllModels, saveCheckpoint, print_parameters, loadCheckpoint
from models.Fed import FedAvg
from utils.helpers import powerset, grangerset, aggListOfDicts, getAllClients

from models.server import server
from models.clients import initClients
from utils.dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple, Gaussian_moment

def train(dataloader, model, loss_fn, optimizer, verbose=False,dp_mechanism,dp_clip,lr,dp_epsilon,dp_delta):
    '''
        Trains a NN model over a dataloader
    '''
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_loss = []
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if dp_mechanism != 'no_dp':
            clip_gradients(dp_mechanism,dp_clip,model)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        batch_loss.append(loss.item())

        if batch % 100 == 0:
            loss, current = sum(batch_loss)/len(batch_loss), batch * len(X)
            if dp_mechanism != 'no_dp':
                add_noise(lr,dp_clip,dp_mechanism,dp_epsilon,dp_delta,model)
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            return loss, model

def clip_gradients(dp_mechanism,dp_clip,net):
    if dp_mechanism == 'Laplace':
            # Laplace use 1 norm
        for k, v in net.named_parameters():
            v.grad /= max(1, v.grad.norm(1) / dp_clip)
    elif dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
        for k, v in net.named_parameters():
            v.grad /= max(1, v.grad.norm(2) / dp_clip)

def add_noise(lr,dp_clip,dp_mechanism,dp_epsilon,dp_delta, net):
    sensitivity = cal_sensitivity(lr, dp_clip, 0)
    if dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Laplace(epsilon=dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise
    elif dp_mechanism == 'Gaussian':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=dp_epsilon, delta=dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(device)
                v += noise



def test(dataloader, model, loss_fn, verbose=False):
    '''
        Tests a NN model over a dataloader
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct, f1 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            f1 += f1_score(y.cpu(), y_pred.argmax(1).cpu(), average='micro')

    test_loss /= num_batches
    correct /= size
    f1 /= num_batches

    if verbose:
        print(f"Test Error: \n Accuracy: {correct:>8f}, Avg loss: {test_loss:>8f}, F1: {f1:>8f} \n")

    return test_loss, correct, f1

def trainClient(client, server,dp_mechanism,dp_clip,lr,dp_epsilon,dp_delta):
    '''
        Trains a client device and saves its parameters
    '''
    # Read client behaviour setting
    client_behaviour = client['behaviour']

    # Load local dataset
    client_dataloader = torch.load("./data/local_train_data.pt")#client['dataloader']

    # Get client model and functions
    client_name = client['name']

    client_model = FederatedModel().to(device)
    client_loss_fn = FederatedLossFunc()
    client_optimizer = FederatedOptimizer(client_model.parameters(), lr=FederatedLearnRate, momentum=FederatedMomentum,
                                          weight_decay=FederatedWeightDecay)

    # If client is adversarial, they return randomized parameters
    if client_behaviour == 'ADVERSARIAL':
        # Save client model state_dicts (simulating client uploading model parameters to server)
        saveCheckpoint(
            client_name,
            client_model.state_dict(),
            client_optimizer.state_dict(),
            client['filepath'],
        )

        test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
        print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

        return 0, test_loss, test_acc, test_f1

    # Load server model state_dicts (simulating client downloading server model parameters)
    checkpoint = loadCheckpoint("./ShapleyValue/train_model/server_model.pt")
    client_model.load_state_dict(checkpoint['model_state_dict'])  # Using current server model parameters
    # client_optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Using current server model parameters

    # If client is a freeloader, they return the same server model parameters
    if client_behaviour == 'FREERIDER':
        # Save client model state_dicts (simulating client uploading model parameters to server)
        saveCheckpoint(
            client_name,
            client_model.state_dict(),
            client_optimizer.state_dict(),
            client['filepath'],
        )

        test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
        print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

        return 0, test_loss, test_acc, test_f1

    # If client is normal, they train client over N epochs
    epochs = args.epoch
    print(f'Training {client_name} over {epochs} epochs...')
    for t in range(epochs):
        train_loss,client_model = train(client_dataloader, client_model, client_loss_fn, client_optimizer,dp_mechanism,dp_clip,lr,dp_epsilon,dp_delta)

    test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
    print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

    # Save client model state_dicts (simulating client uploading model parameters to server)
    saveCheckpoint(
        client_name,
        client_model.state_dict(),
        client_optimizer.state_dict(),
        client['filepath'],
    )

    return train_loss, test_loss, test_acc, test_f1


if __name__ == '__main__':
    # parse args
    args = args_parser()

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
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dp_mechanism=args.dp_mechanism
    dp_clip=args.dp_clip
    lr=args.learning_rate
    dp_epsilon=args.dp_epsilon
    dp_delta=args.dp_delta
    with open('./client.json', 'r') as f:
        client=json.loads(f.read())
    trainClient(client,dp_mechanism,dp_clip,lr,dp_epsilon,dp_delta)


