def FedAvg(model_state_dicts,client_weights):
    '''
        Calculates and generates the FedAvg of the state_dict of a list of models. Returns the FedAvg state_dict.
    '''
    # Sum up tensors from all states
    state_dict_avg = {}  # Stores the sum of state parameters
    for i in range(len(client_weights)):
        state_dict=model_state_dicts[i]
        for key, params in state_dict.items():
            if key in state_dict_avg:
                state_dict_avg[key] += params.detach().clone()*(client_weights[i]/sum(client_weights))
            else:
                state_dict_avg[key] = params.detach().clone()*(client_weights[i]/sum(client_weights))

    # Get Federated Average of clients' parameters
    return state_dict_avg