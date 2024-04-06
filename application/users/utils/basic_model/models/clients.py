
def initClients(cli_num,behaviour_list,client_filepath, dataloaders):
    '''
        Initializes clients objects and returns a list of client object
    '''

    print(f'Initializing client_{cli_num}')
    # Setup client devices

    client_name = f'client_{cli_num}'
    client = {
        'name': client_name,
        'behaviour': behaviour_list[cli_num-1],
        'filepath': f'{client_filepath}{client_name}.pt',
        #'dataloader': dataloaders[cli_num-1]
    }


    print('Client Name / Behaviour:', [(client['name'], client['behaviour']) ], '\n')

    return client

