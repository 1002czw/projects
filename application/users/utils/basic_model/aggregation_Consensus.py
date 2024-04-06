# Load libraries
import math, random, glob, time
from itertools import combinations, permutations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms as T

from sklearn.metrics import f1_score

from utils.parameters import args_parser
from models.Nets import SmallMLP_MNIST, MediumMLP_MNIST, LargeMLP_MNIST, SmallMLP_EMNIST, LargeMLP_EMNIST
from utils.save_file import deleteAllModels, saveCheckpoint, print_parameters, loadCheckpoint
from models.Fed import FedAvg
from utils.helpers import powerset, grangerset, aggListOfDicts, getAllClients
import json

from models.clients import initClients


def train(dataloader, model, loss_fn, optimizer, verbose=False):
    '''
        Trains a NN model over a dataloader
    '''
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            return loss


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

def list_txt(path, list=None):
    '''

    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

def trainClients(clients, server):
    '''
        Trains a list of client devices and saves their parameters
    '''
    loss, acc, f1 = {}, {}, {}
    for client in clients:
        train_loss, test_loss, test_acc, test_f1 = trainClient(client, server)

        # Aggregate statistics
        loss[client['name']] = test_loss
        acc[client['name']] = test_acc
        f1[client['name']] = test_f1

    return loss, acc, f1


def trainClient(client, server):
    '''
        Trains a client device and saves its parameters
    '''
    # Read client behaviour setting
    client_behaviour = client['behaviour']

    # Load local dataset
    client_dataloader = client['dataloader']

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
    checkpoint = loadCheckpoint(server['filepath'])
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
        train_loss = train(client_dataloader, client_model, client_loss_fn, client_optimizer)

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


def evalFedAvg(server):
    '''
        Load client state dicts, perform parameter aggregation and evaluate contributions for each client
    '''
    # Retrieve all clients' uploaded data
    client_filepaths = glob.glob(f"{server['client_filepath']}/client*.pt")

    # Load client model state_dicts (simulating client downloading server model parameters)
    client_checkpoints = []
    for client_filepath in client_filepaths:
        client_checkpoint = loadCheckpoint(client_filepath)
        client_checkpoints += [client_checkpoint]

    # Get Federated Average of clients' parameters
    model_state_dicts = [checkpoint['model_state_dict'] for checkpoint in client_checkpoints]
    fed_model_state_dict = FedAvg(model_state_dicts)

    # Instantiate server model using FedAvg
    fed_model = FederatedModel().to(device)
    fed_model.load_state_dict(fed_model_state_dict)
    fed_model.eval()

    # Evaluate FedAvg server model
    start_time = time.time()  # Time evaluation period
    eval_loss, eval_acc, eval_f1 = test(server['dataloader'], fed_model, server['loss_func'])
    time_taken = time.time() - start_time  # Get model evaluation period (in seconds)
    print(f"\n>> Federated Model Acc: {eval_acc:>8f}, Loss: {eval_loss:>8f}, F1: {eval_f1:>8f}\n")

    # Save server model state_dicts (simulating public access to server model parameters)
    saveCheckpoint(
        server['name'],
        fed_model.state_dict(),
        server['optimizer'],
        server['filepath'],
    )

    # Output statistics
    return eval_loss, eval_acc, eval_f1, time_taken


def trainFedAvgModel(rounds):
    '''
        Train a model using naive FedAvg
    '''

    loss, acc, f1, eval_time = [], [], [], []
    for i in range(rounds):
        print(f'\n=======================\n\tROUND {i + 1}\n=======================')
        clients_loss, clients_acc, clients_f1 = trainClients(clients, server)
        server_loss, server_acc, server_f1, time_taken = evalFedAvg(server)

        # Compile performance measures
        loss += [{**clients_loss, **{'server': server_loss}}]
        acc += [{**clients_acc, **{'server': server_acc}}]
        f1 += [{**clients_f1, **{'server': server_f1}}]
        eval_time += [time_taken]

    # Output statistics
    return aggListOfDicts(loss), aggListOfDicts(acc), aggListOfDicts(f1), eval_time


'''
measurement of contribution using granger causality
'''


def trainFedAvgGrangerModel(rounds=5, granger_filter=True, coalition_limit=0):
    '''
    train a model using FedAvg using granger causality
    '''
    loss, acc, f1, eval_time, best_coalitions, granger = [], [], [], [], [], []
    for i in range(rounds):
        print(f'\n===========\n\tROUND {i + 1}\n=============', coalition_limit)
        clients_loss, clients_acc, clients_f1 = trainClients(clients=clients, server=server)
        start_time = time.time()  # Time evaluation period
        client_granger, coalitions, time_taken = evalFedAvgGranger(server=server, granger_filter=granger_filter,
                                                                   coalition_limit=coalition_limit)
        time_taken = time.time() - start_time  # Accumulate model evaluation period (in seconds)
        print("------time_taken------", time_taken)

        temple = []
        for coalition in coalitions:
            temple.append(client_granger[coalition])
        granger.append(temple)
        eval_time.append(time_taken)

    return granger, eval_time


'''
measurement of contibution using shapley value
'''


def trainFedAvgShapleyModel(rounds=5, shapley_filter=True, coalition_limit=0):
    '''
        Train a model using FedAvg using Shapley Value
    '''

    loss, acc, f1, eval_time, best_coalitions, sv = [], [], [], [], [], []
    for i in range(rounds):
        print(f'\n=======================\n\tROUND {i + 1}\n=======================')

        clients_loss, clients_acc, clients_f1 = trainClients(clients, server)
        start_time = time.time()  # Time evaluation period
        server_loss, server_acc, server_f1, time_taken, best_coalition, clients_sv = evalFedAvgShapley(server,
                                                                                                       shapley_filter,
                                                                                                       coalition_limit)
        time_taken = time.time() - start_time  # Accumulate model evaluation period (in seconds)
        print("------time_taken------", time_taken)

        # Compile performance measures
        loss += [{**clients_loss, **{'server': server_loss}}]
        acc += [{**clients_acc, **{'server': server_acc}}]
        f1 += [{**clients_f1, **{'server': server_f1}}]
        eval_time.append(time_taken)
        best_coalitions += [best_coalition]  # Name of best coalition (string)
        sv += [clients_sv]  # Shapley Values of every client

    # Output statistics
    return aggListOfDicts(loss), aggListOfDicts(acc), aggListOfDicts(f1), eval_time, best_coalitions, aggListOfDicts(sv)


def evalFedAvgGranger(server, granger_filter=True, coalition_limit=0):
    '''
    Evaluates and rewards clinets based on marginal contributions in coalition using granger causality
    '''

    # Evaluate current server model performance
    server_checkpoint = loadCheckpoint(server['filepath'])
    server_model = FederatedModel().to(device)
    server_model.load_state_dict(server_checkpoint['model_state_dict'])

    # Evaluate server model
    server_loss, server_acc, server_f1 = test(server['dataloader'], server_model, server['loss_func'])

    print(f"\n>> Current Server Model Acc: {server_acc:>8f}, Loss: {server_loss:>8f}, F1: {server_f1:>8f}\n")

    # Coalition Limit only valid if Shapley Filter is True
    if not granger_filter:
        coalition_limit = 0

    # Load client model state_dicts (simulating server sideloading client model parameters)
    client_filepaths = glob.glob(f"{server['client_filepath']}/client*.pt")

    client_checkpoints = {}
    for client_filepath in client_filepaths:
        client_checkpoint = loadCheckpoint(client_filepath)
        client_checkpoints[client_checkpoint['name']] = client_checkpoint
    client_names = [client_id for client_id in client_checkpoints]

    # We generate non-null powerset of selected clients
    # coalitions = list([frozenset(subset) for subset in powerset(client_names)])
    coalitions = [subset for subset in grangerset(client_names)]
    allClients = list(getAllClients(client_names))

    # find different client between allClients and each tuple in coalitions
    diff_clients = []

    for coalition in coalitions:
        same_clients = [x for x in list(coalition) if x in allClients]  # 两个列表表都存在
        diff_client = [y for y in (list(coalition) + allClients) if y not in same_clients]  # 两个元组中的不同元素
        diff_clients.extend(diff_client)

    coalitions.append(tuple(allClients))

    # We prune number of coalitions based on Coalition Limit parameter
    if coalition_limit > 0:
        limited_coalitions = []
        for coalition in coalitions:
            if len(coalition) <= coalition_limit:
                limited_coalitions += [coalition]
        coalitions = limited_coalitions

    # We calculate the contributions of each coalition
    print('FedAvg Coalition Evaluations:')

    best_model_state_dict, best_name = server_checkpoint[
                                           'model_state_dict'], 'server'  # We keep track of the best performing model
    best_loss, best_acc, best_f1, best_utility = server_loss, server_acc, server_f1, 0.0
    utilities = {}

    fed_model = FederatedModel().to(device)

    time_taken = 0  # Get model evaluation period (in seconds)
    for coalition in coalitions:
        # Keep track of largest coalition
        is_largest_coalition = False
        if len(coalition) == len(client_names):
            is_largest_coalition = True

        coalition_names = [client_id for client_id in coalition]

        # Get Federated Average of clients' parameters
        model_state_dicts = [client_checkpoints[client_id]['model_state_dict'] for client_id in coalition_names]
        fed_model_state_dict = FedAvg(model_state_dicts)

        # Instantiate server model using FedAvg
        fed_model.load_state_dict(fed_model_state_dict)
        fed_model.eval()

        # Evaluate FedAvg server model
        eval_loss, eval_acc, eval_f1 = test(server['dataloader'], fed_model, FederatedLossFunc())

        print(f">> {'-'.join(coalition_names)} Acc: {eval_acc:>8f}, Loss: {eval_loss:>8f}, F1: {eval_f1:>8f}")

        # Evaluate marginal contributions (Coalition model metrics - Server model metrics) > 0
        utility_loss = max(0, server_loss - eval_loss)  # Reversed because lower loss == better performance
        utility_acc = max(0, eval_acc - server_acc)
        utility_f1 = max(0, eval_f1 - server_f1)

        utility_sum = 0
        for metric in args.reward_metrics:
            if metric == 'LOSS':
                utility_sum += utility_loss
            elif metric == 'ACC':
                utility_sum += utility_acc
            elif metric == 'F1':
                utility_sum += utility_f1

        # Using Shapley Filter, keep the best model and statistics
        if granger_filter:
            if utility_sum > best_utility:
                best_name = '-'.join(coalition_names)
                best_model_state_dict = fed_model_state_dict
                best_utility = utility_sum
                best_loss, best_acc, best_f1 = eval_loss, eval_acc, eval_f1
        # Otherwise, the largest coalition is the best model
        else:
            if is_largest_coalition:
                best_name = '-'.join(coalition_names)
                best_model_state_dict = fed_model_state_dict
                best_utility = utility_sum
                best_loss, best_acc, best_f1 = eval_loss, eval_acc, eval_f1

        utilities[coalition] = {
            'loss': eval_loss,
            'acc': eval_acc,
            'f1': eval_f1,
            'sum': utility_sum
        }

    print(f'\nFedAvg Coalition Contribution using Granger Causality')
    all_loss = utilities[coalitions[len(coalitions) - 1]]['loss']
    coalitions.pop()

    client_granger = {}
    i = 0

    for coalition in coalitions:
        client_granger[coalition] = {
            'client_name': diff_clients[i],
            'granger_value': utilities[coalition]['loss'] - all_loss
        }
        i += 1

    return client_granger, coalitions, time_taken


def judgeConsensus(clientSV, contribution):
    if contribution != {}:
        if len(contribution) == len(clientSV):
            result_list=[]
            for k in contribution:
                result_list.append(abs(clientSV[k] - contribution[k][args.reward_metrics]))
            if max(result_list)< args.win_condition:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def evalFedAvgShapley(server, shapley_filter=True, coalition_limit=0):
    '''
        Evaluates and rewards clients based on marginal contributions in coalition permutations
    '''

    # Evaluate current server model performance
    server_checkpoint = loadCheckpoint(f'./{args.loadpath}/train_model/server_model.pt')
    server_model = FederatedModel().to(device)
    server_model.load_state_dict(server_checkpoint['model_state_dict'])

    test_dataloader = torch.load(f'./{args.loadpath}/train_model/test_data.pt')

    # Evaluate server model
    server_loss, server_acc, server_f1 = test(test_dataloader, server_model, FederatedLossFunc())


    print(f"\n>> Current Global Model Acc: {server_acc:>8f}, Loss: {server_loss:>8f}, F1: {server_f1:>8f}\n")

    # Coalition Limit only valid if Shapley Filter is True
    if not shapley_filter:
        coalition_limit = 0

    # Load client model state_dicts (simulating server sideloading client model parameters)
    client_filepaths = glob.glob(f"../global_Model/global_Model_Server/{args.loadpath}/client/client*.pt")

    client_checkpoints = {}
    for client_filepath in client_filepaths:
        client_checkpoint = loadCheckpoint(client_filepath)
        client_checkpoints[client_checkpoint['name']] = client_checkpoint
    client_names = [client_id for client_id in client_checkpoints]

    # We generate non-null powerset of selected clients
    coalitions = list([frozenset(subset) for subset in powerset(client_names)])

    # We prune number of coalitions based on Coalition Limit parameter
    if coalition_limit > 0:
        limited_coalitions = []
        for coalition in coalitions:
            if len(coalition) <= coalition_limit:
                limited_coalitions += [coalition]
        coalitions = limited_coalitions

    # We generate order permutations of selected clients
    #orders = list(permutations(client_names))

    # We prune length of coalitions based on Coalition Limit parameter
    #if coalition_limit > 0:
    #orders = list(set([order[:coalition_limit] for order in orders]))

    # We calculate the contributions of each coalition
    fed_model = FederatedModel().to(device)
    print('FedAvg Coalition Evaluations and Contribution:')
    #cli_num=args.clinum

    with open('./client_fraction.json', 'r') as f:
        fraction_dict=json.loads(f.read())
    if args.num_of_consensus_oneround==1:
        start_time = time.time()  # Time evaluation period
        utilities = {}
        time_taken = 0  # Get model evaluation period (in seconds)
        sv_time = 0
        rNum = random.randint(1, 3)
        contributions = {}
        path_sv = f'../global_Model/global_Model_Server/{args.loadpath}/cliEval.json'
        with open(path_sv, 'r') as f:
            client_sv = json.loads(f.read())
        judge = judgeConsensus(client_sv, contributions)
        while judge == False:
            while sv_time < rNum and args.communications == "delay":
                #Num_permu = random.randint(1, len(client_names)-1)
                #tem_list =[my_name]+random.sample(my_candidate_client_names, Num_permu)
                Num_permu = random.randint(1, len(client_names))
                tem_list =random.sample(client_names, Num_permu)
                order = [client_id for client_id in tem_list]
                index = 1
                prev_suborder = []
                old_loss = server_loss
                old_acc = server_acc
                old_f1 = server_f1
                # Calculate contribution of each client in this order
                for client_id in order:

                    cur_suborder = order[:index]  # eg. ['A'] -> ['A','B'] -> ['A','B','C']
                    # If index > 1, we deduct this suborder's utility from prev suborder (eg. u(AB) - u(A) = c(B))
                    if index > 1:
                        model_state_dicts = [client_checkpoints[client_id]['model_state_dict'] for client_id in
                                             cur_suborder]
                        client_weights=[fraction_dict[client_id] for client_id in cur_suborder]
                        fed_model_state_dict = FedAvg(model_state_dicts,client_weights)

                        # Instantiate server model using FedAvg
                        fed_model.load_state_dict(fed_model_state_dict)
                        fed_model.eval()

                        # Evaluate FedAvg server model
                        eval_loss, eval_acc, eval_f1 = test(test_dataloader, fed_model, FederatedLossFunc())
                        utility_loss = max(0, old_loss - eval_loss)  # Reversed because lower loss == better performance
                        utility_acc = max(0, eval_acc - old_acc)
                        utility_f1 = max(0, eval_f1 - old_f1)

                        old_loss=eval_loss
                        old_acc=eval_acc
                        old_f1=eval_f1

                        utility_sum = 0
                        utility_sum = utility_loss + utility_acc + utility_f1

                        utilities[frozenset(cur_suborder)] = {
                            'loss': utility_loss,
                            'acc': utility_acc,
                            'f1': utility_f1,
                            'sum': utility_sum
                        }
                        cur_utilities = utilities[frozenset(cur_suborder)]
                        prev_utilities = utilities[frozenset(prev_suborder)]
                        ans = {}
                        for utility_metric in cur_utilities:
                            ans[utility_metric] = max(0, cur_utilities[utility_metric] - prev_utilities[utility_metric])

                    # If index == 1, this is a single element's contribution (eg. u(A) = c(A))
                    else:
                        model_state_dicts = [client_checkpoints[client_id]['model_state_dict']]
                        client_weights=[fraction_dict[client_id] for client_id in cur_suborder]
                        fed_model_state_dict = FedAvg(model_state_dicts,client_weights)

                        # Instantiate server model using FedAvg
                        fed_model.load_state_dict(fed_model_state_dict)
                        fed_model.eval()

                        # Evaluate FedAvg server model
                        eval_loss, eval_acc, eval_f1 = test(test_dataloader, fed_model, FederatedLossFunc())
                        utility_loss = max(0,
                                           old_loss - eval_loss)  # Reversed because lower loss == better performance
                        utility_acc = max(0, eval_acc - old_acc)
                        utility_f1 = max(0, eval_f1 - old_f1)

                        old_loss=eval_loss
                        old_acc=eval_acc
                        old_f1=eval_f1

                        utility_sum = 0
                        utility_sum = utility_loss + utility_acc + utility_f1

                        utilities[frozenset([client_id])] = {
                            'loss': utility_loss,
                            'acc': utility_acc,
                            'f1': utility_f1,
                            'sum': utility_sum
                        }

                        # Evaluate marginal contributions (Coalition model metrics - Server model metrics) > 0

                        ans = utilities[frozenset([client_id])]  # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素

                    # Add value to client's list of contributions
                    if not client_id in contributions:
                        contributions[client_id] = {}

                    for utility_metric in ans:
                        if utility_metric in contributions[client_id]:
                            contributions[client_id][utility_metric] = (contributions[client_id][utility_metric] + ans[
                                utility_metric])
                        else:
                            contributions[client_id][utility_metric] = ans[utility_metric]

                    index += 1
                    prev_suborder += [client_id]
                sv_time += 1

            while sv_time >= rNum and judge == False:
                tem_list =random.sample(client_names, len(client_names))
                #random.sample(client_names, len(client_names))
                order = tuple([client_id for client_id in tem_list])
                index = 1
                prev_suborder = []
                old_loss = server_loss
                old_acc = server_acc
                old_f1 = server_f1
                # Calculate contribution of each client in this order
                for client_id in order:

                    cur_suborder = order[:index]  # eg. ['A'] -> ['A','B'] -> ['A','B','C']
                    # If index > 1, we deduct this suborder's utility from prev suborder (eg. u(AB) - u(A) = c(B))
                    if index > 1:
                        model_state_dicts = [client_checkpoints[client_id]['model_state_dict'] for client_id in
                                             cur_suborder]
                        client_weights=[fraction_dict[client_id] for client_id in cur_suborder]
                        fed_model_state_dict = FedAvg(model_state_dicts,client_weights)

                        # Instantiate server model using FedAvg
                        fed_model.load_state_dict(fed_model_state_dict)
                        fed_model.eval()

                        # Evaluate FedAvg server model
                        eval_loss, eval_acc, eval_f1 = test(test_dataloader, fed_model, FederatedLossFunc())
                        utility_loss = max(0, old_loss - eval_loss)  # Reversed because lower loss == better performance
                        utility_acc = max(0, eval_acc - old_acc)
                        utility_f1 = max(0, eval_f1 - old_f1)

                        old_loss=eval_loss
                        old_acc=eval_acc
                        old_f1=eval_f1

                        utility_sum = 0
                        utility_sum = utility_loss + utility_acc + utility_f1

                        utilities[frozenset(cur_suborder)] = {
                            'loss': utility_loss,
                            'acc': utility_acc,
                            'f1': utility_f1,
                            'sum': utility_sum
                        }
                        cur_utilities = utilities[frozenset(cur_suborder)]
                        prev_utilities = utilities[frozenset(prev_suborder)]
                        ans = {}
                        for utility_metric in cur_utilities:
                            ans[utility_metric] = max(0, cur_utilities[utility_metric] - prev_utilities[utility_metric])

                    # If index == 1, this is a single element's contribution (eg. u(A) = c(A))
                    else:
                        model_state_dicts = [client_checkpoints[client_id]['model_state_dict']]
                        client_weights=[fraction_dict[client_id] for client_id in cur_suborder]
                        fed_model_state_dict = FedAvg(model_state_dicts,client_weights)

                        # Instantiate server model using FedAvg
                        fed_model.load_state_dict(fed_model_state_dict)
                        fed_model.eval()

                        # Evaluate FedAvg server model
                        eval_loss, eval_acc, eval_f1 = test(test_dataloader, fed_model, FederatedLossFunc())
                        utility_loss = max(0,
                                           old_loss - eval_loss)  # Reversed because lower loss == better performance
                        utility_acc = max(0, eval_acc - old_acc)
                        utility_f1 = max(0, eval_f1 - old_f1)

                        old_loss=eval_loss
                        old_acc=eval_acc
                        old_f1=eval_f1

                        utility_sum = 0
                        utility_sum = utility_loss + utility_acc + utility_f1

                        utilities[frozenset([client_id])] = {
                            'loss': utility_loss,
                            'acc': utility_acc,
                            'f1': utility_f1,
                            'sum': utility_sum
                        }

                        # Evaluate marginal contributions (Coalition model metrics - Server model metrics) > 0

                        ans = utilities[frozenset([client_id])]  # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素

                    # Add value to client's list of contributions
                    if not client_id in contributions:
                        contributions[client_id] = {}

                    for utility_metric in ans:
                        if utility_metric in contributions[client_id]:
                            contributions[client_id][utility_metric] = (contributions[client_id][utility_metric] + ans[
                                utility_metric])
                        else:
                            contributions[client_id][utility_metric] = ans[utility_metric]

                    index += 1
                    prev_suborder += [client_id]

                sv_time += 1

                if sv_time % 3 == 0:
                    with open(path_sv, 'r') as f:
                        client_sv = json.loads(f.read())

                    for cliID in contributions:
                        for utility_metric in contributions[cliID]:
                            contributions[cliID][utility_metric]=contributions[cliID][utility_metric] / sv_time

                    judge = judgeConsensus(client_sv, contributions)
                    judge_Out = judge
                    judge = True


    else:
        path_sv = f'../global_Model/global_Model_Server/{args.loadpath}/cliEval.json'
        with open(path_sv, 'r') as f:
            client_sv = json.loads(f.read())
        path_sv_cli = f'../global_Model/global_Model_Server/{args.loadpath}/eval_out/eval_out_static_cli_{args.clinum}.json'
        with open(path_sv_cli, 'r') as f:
            client_sv_last = json.loads(f.read())
        start_time = time.time()
        sv_time = client_sv_last["sv_time"]
        judge = client_sv_last["judge_Out"]
        contributions = client_sv_last["Contributions"]
        for cliID in contributions:
            for utility_metric in contributions[cliID]:
                contributions[cliID][utility_metric]=contributions[cliID][utility_metric] * sv_time
        sv_time=sv_time+1
        newcontributions={}
        while judge == False:
            old_loss = server_loss
            old_acc = server_acc
            old_f1 = server_f1
            utilities = {}
            fed_model = FederatedModel().to(device)
            time_taken1 = 0  # Get model evaluation period (in seconds)
            tem_list = random.sample(client_names, len(client_names))
            #tem_list =[my_name]+random.sample(my_candidate_client_names, len(my_candidate_client_names))
            order = tuple([client_id for client_id in tem_list])
            index = 1
            prev_suborder = []
            # Calculate contribution of each client in this order
            for client_id in order:

                cur_suborder = order[:index]  # eg. ['A'] -> ['A','B'] -> ['A','B','C']
                # If index > 1, we deduct this suborder's utility from prev suborder (eg. u(AB) - u(A) = c(B))
                if index > 1:
                    model_state_dicts = [client_checkpoints[client_id]['model_state_dict'] for client_id in
                                         cur_suborder]
                    client_weights=[fraction_dict[client_id] for client_id in cur_suborder]
                    fed_model_state_dict = FedAvg(model_state_dicts,client_weights)

                    # Instantiate server model using FedAvg
                    fed_model.load_state_dict(fed_model_state_dict)
                    fed_model.eval()

                    # Evaluate FedAvg server model
                    eval_loss, eval_acc, eval_f1 = test(test_dataloader, fed_model, FederatedLossFunc())
                    utility_loss = max(0, old_loss - eval_loss)  # Reversed because lower loss == better performance
                    utility_acc = max(0, eval_acc - old_acc)
                    utility_f1 = max(0, eval_f1 - old_f1)
                    old_loss=eval_loss
                    old_acc=eval_acc
                    old_f1=eval_f1

                    utility_sum = 0
                    utility_sum = utility_loss + utility_acc + utility_f1
                    utilities[frozenset(cur_suborder)] = {
                        'loss': utility_loss,
                        'acc': utility_acc,
                        'f1': utility_f1,
                        'sum': utility_sum
                    }
                    cur_utilities = utilities[frozenset(cur_suborder)]
                    prev_utilities = utilities[frozenset(prev_suborder)]
                    ans = {}
                    for utility_metric in cur_utilities:
                        ans[utility_metric] = max(0, cur_utilities[utility_metric] - prev_utilities[utility_metric])

                # If index == 1, this is a single element's contribution (eg. u(A) = c(A))
                else:
                    model_state_dicts = [client_checkpoints[client_id]['model_state_dict']]
                    client_weights=[fraction_dict[client_id] for client_id in cur_suborder]
                    fed_model_state_dict = FedAvg(model_state_dicts,client_weights)

                    # Instantiate server model using FedAvg
                    fed_model.load_state_dict(fed_model_state_dict)
                    fed_model.eval()

                    # Evaluate FedAvg server model
                    eval_loss, eval_acc, eval_f1 = test(test_dataloader, fed_model, FederatedLossFunc())
                    utility_loss = max(0, old_loss - eval_loss)  # Reversed because lower loss == better performance
                    utility_acc = max(0, eval_acc - old_acc)
                    utility_f1 = max(0, eval_f1 - old_f1)
                    old_loss=eval_loss
                    old_acc=eval_acc
                    old_f1=eval_f1

                    utility_sum = utility_loss + utility_acc + utility_f1
                    utilities[frozenset([client_id])] = {
                        'loss': utility_loss,
                        'acc': utility_acc,
                        'f1': utility_f1,
                        'sum': utility_sum
                    }

                    # Evaluate marginal contributions (Coalition model metrics - Server model metrics) > 0
                    ans = utilities[frozenset([client_id])]  # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素

                # Add value to client's list of contributions
                if not client_id in newcontributions:
                    newcontributions[client_id] = {}
                for utility_metric in ans:
                    if utility_metric in newcontributions[client_id]:
                        newcontributions[client_id][utility_metric] = (newcontributions[client_id][utility_metric] + ans[
                            utility_metric])
                    else:
                        newcontributions[client_id][utility_metric] = ans[utility_metric]
                index += 1
                prev_suborder += [client_id]
            sv_time += 1
            if sv_time % 3 == 0:
                with open(path_sv, 'r') as f:
                    client_sv = json.loads(f.read())
                for cliID in contributions:
                    for utility_metric in contributions[cliID]:
                        contributions[cliID][utility_metric]=(contributions[cliID][utility_metric]+newcontributions[cliID][utility_metric]) / sv_time
                judge = judgeConsensus(client_sv, contributions)
                judge_Out = judge
                judge = True

    '''
        Load client state dicts, perform parameter aggregation and evaluate contributions for each client
    '''

    # We calculate the contributions of each client in order using utility
    # We calculate the Shapley Value of each client by averaging the sum of their contributions
    print(f'\nClient_{args.clinum} Compute Shapley Values Results:')
    for client_id in contributions:
        txt = f'>> {client_id}:'
        for metric in contributions[client_id]:
            txt += f' {metric}: {contributions[client_id][metric]:>8f},'
        print(txt)

    # Output statistics
    if args.num_of_consensus_oneround == 1:
        time_taken = time.time() - start_time
    else:
        time_taken = time.time() - start_time + client_sv_last["Time"]  # Accumulate model evaluation period (in seconds)
    return judge_Out, time_taken, sv_time, contributions


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


if __name__ == '__main__':
    # parse args
    args = args_parser()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download training and test data from open datasets
    # MLP model uses Fashion-MNIST
    if args.dataset_type == 'MNIST':
        train_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=T.ToTensor(),
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=T.ToTensor(),
        )

    elif args.dataset_type == 'EMNIST':
        train_data = datasets.EMNIST(
            root="data",
            train=True,
            download=True,
            split='balanced',
            transform=T.ToTensor(),
        )

        test_data = datasets.EMNIST(
            root="data",
            train=False,
            download=True,
            split='balanced',
            transform=T.ToTensor(),
        )

    # Define network model architecture
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

    if args.train_type == 'naive':

        #  evaluate and consensus
        fedavg_loss, fedavg_acc, fedavg_f1, fedavg_time = trainFedAvgModel(args.common_rounds)
    elif args.train_type == 'FedAvg_Shapley':

        #  evaluate and consensus
        # sv_loss, sv_acc, sv_f1, sv_time, sv_best_coalitions, sv = trainFedAvgShapleyModel(rounds=args.common_rounds, shapley_filter=args.shapley_filter, coalition_limit=args.coalition_limit)
        server = torch.load(f'./{args.loadpath}/train_model/server_model.pt')
        judge_Out, time_taken, sv_time, sv = evalFedAvgShapley(server, shapley_filter=args.shapley_filter,
                                                               coalition_limit=args.coalition_limit)

        out_Static = {
            "Name": f'Client_{args.clinum}',
            "judge_Out": judge_Out,
            "Time": time_taken,
            "sv_time": sv_time,
            "Contributions": sv,
        }
        json_str = json.dumps(out_Static)
        with open(
                f'../global_Model/global_Model_Server/{args.loadpath}/eval_out/eval_out_static_cli_{args.clinum}.json',
                'w') as json_file:
            json_file.write(json_str)

    elif args.train_type == 'FedAvg_Granger':

        #  evaluate and consensus
        granger, eval_time = trainFedAvgGrangerModel(rounds=args.common_rounds, granger_filter=args.shapley_filter,
                                                     coalition_limit=args.coalition_limit)
        print("---eval time-----", eval_time)
