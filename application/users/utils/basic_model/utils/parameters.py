import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Federated network settings
    parser.add_argument('--model_size', type=str, default='MEDIUM', help="the type of model small/medium/large")
    parser.add_argument('--reward_metrics', nargs='+', type=str, default='f1', help="decide the contribution evaluation metric:loss, acc, f1")
    parser.add_argument('--common_rounds', type=int, default=15, help="the training round")
    parser.add_argument('--shapley_filter', type=bool, default=True, help='True / False - If true, select only the best coalition model per communication round')
    parser.add_argument('--coalition_limit', type=int, default=0, help='Limits the size of each coalitions (Set as non-positive number to disable limit) or the limitation of length of each coalition')

    # Training dataset settings
    parser.add_argument('--dataset_type', type=str, default='MNIST', help='MNIST / EMNIST')
    parser.add_argument('--distribution_type', type=str, default='IID', help='IID / NIID_1 / NIID_2 / NIID_12')
    parser.add_argument('--batchsize', type=int, default=64, help='Dataset batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--clinum', type=int, default=1, help='client number')
    parser.add_argument('--loadpath', type=str, default='ShapleyValue', help='epoch')

    # Client behaviours each parameter represents the number of clients running in the network
    parser.add_argument('--num_normal_clients', type=int, default=50, help='Client trains model and returns updated parameters')
    parser.add_argument('--num_freerider_clients', type=int, default=0, help='Client does not train model and returns original parameters')
    parser.add_argument('--num_adversarial_clients', type=int, default=0, help='Client returns randomized parameters')

    # train
    parser.add_argument('--train_type', type=str, default='FedAvg_Shapley', help='Training the type of model')

    #privacy-DP parameters
    parser.add_argument('--dp_mechanism', type=str, default='no_dp', help='Client train models using dp mechanism')
    parser.add_argument('--dp_clip', type=float, default=20, help='dp mechanism parameters')
    parser.add_argument('--dp_epsilon', type=float, default=20, help='dp algorithm arameters')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='Gaussian dp algorithm parameters')

    #communications
    parser.add_argument('--communications', type=str, default='delay', help='Help judge the communications conditions to decide the permutation:good, delay, bad')
    parser.add_argument('--win_condition', type=float, default=0.01, help='Help judge who is the successful miner')
    parser.add_argument('--metric', type=float, default=0.0005, help='Help judge who is the successful miner')
    parser.add_argument('--num_of_consensus_oneround', type=int, default=1, help='Help judge the times of consensus at one round')

    args = parser.parse_args()
    return args
