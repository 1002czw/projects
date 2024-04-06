# Functions to split dataset based on IID or Non-IID selection
import random
import numpy as np

def avg_divide(l, g):

    #将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
    #每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
    #返回由不同的groups组成的列表
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_idcs(l, idcs):
    """
    将列表`l` 划分为长度为 `len(idcs)` 的子列表
    第`i`个子列表从下标 `idcs[i]` 到下标`idcs[i+1]`
    （从下标0到下标`idcs[0]`的子列表另算）
    返回一个由多个子列表组成的列表
    """
    res = []
    current_index = 0
    for index in idcs:
        res.append(l[current_index: index])
        current_index = index

    return res

def prepareIID(dataset, num_clients):
    '''
        Prepares IID training datasets for each client
    '''
    dataset_split = [[] for i in range(num_clients)]

    for idx, sample in enumerate(dataset):
        dataset_split[idx % num_clients] += [sample]

    return dataset_split

def prepareNIID1(dataset, num_clients):
    '''
        Prepares NIID-1 training datasets for each client ----Different Distributions and Same Size
        Each participant has the same number of samples. However, participant 1 & 2’s datasets contain 80% of digits ’1’ and ’2’. The other digits evenly divide the remaining 20% of the samples. Similar procedures are applied to the rest of the participants.
    '''
    #dataset_split = [[] for i in range(num_clients)]
    n_class=len(dataset.classes)
    all_labels = list(range(n_class))
    np.random.shuffle(all_labels)

    clusters_labels = avg_divide(all_labels, n_class)
    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    data_idcs = list(range(len(dataset)))
    clusters_sizes = np.zeros(n_class, dtype=int)
    # 存储每个cluster对应的数据索引
    clusters = {k: [] for k in range(n_class)}
    for idx in data_idcs:
        _, label = dataset[idx]
        # 由样本数据的label先找到其cluster的id
        group_id = label2cluster[label]
        # 再将对应cluster的大小+1
        clusters_sizes[group_id] += 1
        # 将样本索引加入其cluster对应的列表中
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        random.shuffle(cluster)
    # 记录某个cluster的样本分到某个client上的数量
    clients_counts = np.zeros((n_class, num_clients), dtype=np.int64)
    '''
    num_rest_per=0.2/(num_clients-2)
    alpha=num_rest_per* np.ones(num_clients)
    t=0
    # 遍历每一个cluster
    for cluster_id in range(n_class):
        if t % 2 !=0:
                alpha=num_rest_per* np.ones(num_clients)
        for w_i in range(len(alpha)):

            if w_i==cluster_id:
                t=w_i
                alpha[w_i]=0.4

        if t % 2==0 and t < num_clients-1:
            alpha[t+1]=0.4

        print(alpha)
    '''
    '''
    alpha=[[0.5,0.125,0.125,0.125,0.125],
           [0.5,0.125,0.125,0.125,0.125],
           [0.125,0.5,0.125,0.125,0.125],
           [0.125,0.5,0.125,0.125,0.125],
           [0.125,0.125,0.5,0.125,0.125],
           [0.125,0.125,0.5,0.125,0.125],
           [0.125,0.125,0.125,0.5,0.125],
           [0.125,0.125,0.125,0.5,0.125],
           [0.125,0.125,0.125,0.125,0.5],
           [0.125,0.125,0.125,0.125,0.5]]
    '''
    alpha=[[0.4,0.4,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025],
           [0.4,0.4,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025],
           [0.025,0.025,0.4,0.4,0.025,0.025,0.025,0.025,0.025,0.025],
           [0.025,0.025,0.4,0.4,0.025,0.025,0.025,0.025,0.025,0.025],
           [0.025,0.025,0.025,0.025,0.4,0.4,0.025,0.025,0.025,0.025],
           [0.025,0.025,0.025,0.025,0.4,0.4,0.025,0.025,0.025,0.025],
           [0.025,0.025,0.025,0.025,0.025,0.025,0.4,0.4,0.025,0.025],
           [0.025,0.025,0.025,0.025,0.025,0.025,0.4,0.4,0.025,0.025],
           [0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.4,0.4],
           [0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.4,0.4],]
    for cluster_id in range(n_class):

        weight= alpha[cluster_id]*np.ones(num_clients)

        # 对每个client赋予一个满足dirichlet分布的权重，用于该cluster样本的分
        # np.random.multinomial 表示投掷骰子clusters_sizes[cluster_id](该cluster中的样本数)次，落在各client上的权重依次是weights
        # 该函数返回落在各client上各多少次，也就对应着各client应该分得来自该cluster的样本数
        clients_counts[cluster_id] = clusters_sizes[cluster_id]*weight

    # 对每一个cluster上的每一个client的计数次数进行前缀（累加）求和，
    # 相当于最终返回的是每一个cluster中按照client进行划分的样本分界点下标
    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_idcs = [[] for _ in range(num_clients)]
    for cluster_id in range(n_class):
        # cluster_split为一个cluster中按照client划分好的样本
        cluster_split = split_list_by_idcs(clusters[cluster_id], clients_counts[cluster_id])

        # 将每一个client的样本累加上去
        for client_id, idcs in enumerate(cluster_split):

            clients_idcs[client_id] += idcs



    dataset_split = [[] for i in range(num_clients)]
    for cli_id in range(num_clients):
        list_data=[dataset[idx] for idx in clients_idcs[cli_id]]
        random.shuffle(list_data)
        dataset_split[cli_id] += list_data



    return dataset_split


def prepareNIID2(dataset, num_clients):
    '''
        Prepares NIID-1 training datasets for each client-----Same Distribution and Different Sizes
    '''
    dataset_split = [[] for i in range(num_clients)]
    alpha=[0.05,0.05,0.075,0.075,0.1,0.1,0.125,0.125,0.15,0.15] * np.ones(num_clients)
    #alpha=[0.1,0.15,0.2,0.25,0.3]* np.ones(num_clients)
    for i in range(num_clients):
        index1=sum(alpha[:i])*len(dataset)
        index2=sum(alpha[:i+1])*len(dataset)
        for idx, sample in enumerate(dataset):
            if index1 <= idx <=index2-1:
                dataset_split[i] += [sample]

    return dataset_split


def prepareNIID12(dataset, num_clients):

    n_class=len(dataset.classes)
    all_labels = list(range(n_class))
    np.random.shuffle(all_labels)

    clusters_labels = avg_divide(all_labels, n_class)
    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    data_idcs = list(range(len(dataset)))
    clusters_sizes = np.zeros(n_class, dtype=int)
    # 存储每个cluster对应的数据索引
    clusters = {k: [] for k in range(n_class)}
    for idx in data_idcs:
        _, label = dataset[idx]
        # 由样本数据的label先找到其cluster的id
        group_id = label2cluster[label]
        # 再将对应cluster的大小+1
        clusters_sizes[group_id] += 1
        # 将样本索引加入其cluster对应的列表中
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        random.shuffle(cluster)
    # 记录某个cluster的样本分到某个client上的数量
    clients_counts = np.zeros((n_class, num_clients), dtype=np.int64)


    # 遍历每一个cluster
    alpha=[0.05,0.05,0.075,0.075,0.1,0.1,0.125,0.125,0.15,0.15] * np.ones(num_clients)
    for cluster_id in range(n_class):

        # 对每个client赋予一个满足dirichlet分布的权重，用于该cluster样本的分
        # np.random.multinomial 表示投掷骰子clusters_sizes[cluster_id](该cluster中的样本数)次，落在各client上的权重依次是weights
        # 该函数返回落在各client上各多少次，也就对应着各client应该分得来自该cluster的样本数
        clients_counts[cluster_id] = (clusters_sizes[cluster_id] * alpha)

    # 对每一个cluster上的每一个client的计数次数进行前缀（累加）求和，
    # 相当于最终返回的是每一个cluster中按照client进行划分的样本分界点下标
    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_idcs = [[] for _ in range(num_clients)]
    for cluster_id in range(n_class):
        # cluster_split为一个cluster中按照client划分好的样本
        cluster_split = split_list_by_idcs(clusters[cluster_id], clients_counts[cluster_id])

        # 将每一个client的样本累加上去
        for client_id, idcs in enumerate(cluster_split):

            clients_idcs[client_id] += idcs



    dataset_split = [[] for i in range(num_clients)]
    for cli_id in range(num_clients):
        list_data=[dataset[idx] for idx in clients_idcs[cli_id]]
        random.shuffle(list_data)
        dataset_split[cli_id] += list_data

    return dataset_split
