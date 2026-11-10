import torch
import numpy as np
import torch.utils.data
import os
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
from torch.utils.data import DataLoader, Subset
import scipy.stats as stats
import math
# statistic, pvalue = stats.ks_2samp(client1_data, client2_data)
import random
def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        # data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return scaler

def split_data_by_days(data, val_days, test_days, interval=30):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    x=-T * int(test_days)
    test_data = data[-T*int(test_days):]
    val_data = data[-T*int(test_days + val_days): -T*int(test_days)]
    train_data = data[:-T*int(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorFloat = torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def flatten_data(x):
    return x.reshape(-1)  # flatten B and L

def split_node_indices(indices, num_clients, client_node_ratio=None):
    data_len = len(indices)
    ratios = client_node_ratio
    if isinstance(ratios, str):
        ratios = [float(i) for i in ratios.split(',') if i.strip()]

    if not ratios:
        split_size = data_len // num_clients
        remainder = data_len % num_clients
        return [
            indices[i * split_size + min(i, remainder): (i + 1) * split_size + min(i + 1, remainder)]
            for i in range(num_clients)
        ]

    if len(ratios) != num_clients:
        raise ValueError('client_node_ratio length must equal num_clients')
    if any(r < 0 for r in ratios):
        raise ValueError('client_node_ratio must be non-negative')

    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError('client_node_ratio sum must be positive')

    normalized = [r / ratio_sum for r in ratios]
    raw_counts = [r * data_len for r in normalized]
    counts = [int(math.floor(c)) for c in raw_counts]
    remainder = data_len - sum(counts)
    remainder_order = sorted(
        range(num_clients),
        key=lambda i: raw_counts[i] - counts[i],
        reverse=True,
    )
    for i in remainder_order[:remainder]:
        counts[i] += 1

    if any(c == 0 for c in counts):
        raise ValueError('client_node_ratio creates a client with 0 nodes')

    client_indices = []
    start = 0
    for count in counts:
        client_indices.append(indices[start:start + count])
        start += count
    return client_indices

def dtw_distance(series_a, series_b, window):
    n = len(series_a)
    m = len(series_b)
    window = max(window, abs(n - m))
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, n + 1):
        start = max(1, i - window)
        end = min(m, i + window)
        curr.fill(np.inf)
        for j in range(start, end + 1):
            cost = abs(series_a[i - 1] - series_b[j - 1])
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev
    return prev[m]

def prepare_dtw_series(data, max_len):
    series = data[..., 0]
    if series.shape[0] > max_len:
        sample_idx = np.linspace(0, series.shape[0] - 1, max_len).astype(np.int64)
        series = series[sample_idx]
    series = series.astype(np.float64).T
    mean = series.mean(axis=1, keepdims=True)
    std = series.std(axis=1, keepdims=True)
    return (series - mean) / (std + 1e-6)

def get_dtw_cache_path(args, num_nodes):
    cache_dir = getattr(args, 'dtw_cache_dir', './data/.dtw_cache')
    os.makedirs(cache_dir, exist_ok=True)
    dataset = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in args.dataset)
    series_len = getattr(args, 'dtw_series_len', 256)
    window = getattr(args, 'dtw_window', 24)
    return os.path.join(cache_dir, f'{dataset}_N{num_nodes}_L{series_len}_W{window}_dtw.npy')

def compute_dtw_distance_matrix(data, args):
    series = prepare_dtw_series(data, getattr(args, 'dtw_series_len', 256))
    num_nodes = series.shape[0]
    cache_path = get_dtw_cache_path(args, num_nodes)
    if os.path.exists(cache_path):
        print('Load DTW distance cache: ', cache_path)
        return np.load(cache_path)

    window = getattr(args, 'dtw_window', 24)
    dist = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        if i % 10 == 0:
            print(f'Computing DTW graph: node {i}/{num_nodes}')
        for j in range(i + 1, num_nodes):
            d = dtw_distance(series[i], series[j], window)
            dist[i, j] = d
            dist[j, i] = d

    np.save(cache_path, dist)
    print('Save DTW distance cache: ', cache_path)
    return dist

def build_knn_graph_from_distance(dist, topk):
    num_nodes = dist.shape[0]
    neighbors = [set() for _ in range(num_nodes)]
    topk = min(topk, num_nodes - 1)
    for i in range(num_nodes):
        nearest = np.argsort(dist[i])[1:topk + 1]
        for j in nearest:
            neighbors[i].add(int(j))
            neighbors[int(j)].add(i)

    adjacency = []
    eweights = []
    positive_dist = dist[dist > 0]
    scale = float(np.median(positive_dist)) if positive_dist.size else 1.0
    for i in range(num_nodes):
        node_neighbors = sorted(neighbors[i])
        adjacency.append(node_neighbors)
        weights = []
        for j in node_neighbors:
            sim = scale / (dist[i, j] + 1e-6)
            weights.append(min(1000000, max(1, int(round(sim * 1000)))))
        eweights.append(weights)
    return adjacency, eweights

def split_node_indices_by_dtw_metis(data, args, num_clients):
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError('pymetis is required for client_partition_method=metis_dtw') from exc

    dist = compute_dtw_distance_matrix(data[..., :getattr(args, 'input_dim', 1)], args)
    adjacency, eweights = build_knn_graph_from_distance(dist, getattr(args, 'dtw_topk', 8))
    xadj = [0]
    adjncy = []
    flat_eweights = []
    for node_neighbors, node_weights in zip(adjacency, eweights):
        adjncy.extend(node_neighbors)
        flat_eweights.extend(node_weights)
        xadj.append(len(adjncy))
    _, membership = pymetis.part_graph(
        num_clients,
        xadj=xadj,
        adjncy=adjncy,
        eweights=flat_eweights,
    )

    client_indices = [[] for _ in range(num_clients)]
    for node_id, part_id in enumerate(membership):
        client_indices[int(part_id)].append(node_id)

    if any(len(i) == 0 for i in client_indices):
        raise ValueError('METIS produced an empty client partition')
    return client_indices

def get_dataloader(args, num_clients, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    # data = load_st_dataset(args.dataset)        # B, N, D
    if args.dataset in ['PEMS-BAY']:

        df = load_st_dataset(args.dataset)
        #normalize st data
        # scaler = normalize_dataset(data, normalizer, args.column_wise)

        # num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        # feature_list = [data]

        L, N, F = data.shape

        feature_list = [data]

        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

        day_in_week = df.index.dayofweek
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)
    else:
        # load raw st dataset
        data = load_st_dataset(args.dataset)  # B, N, D
        # normalize st data
        # scaler = normalize_dataset(data, normalizer, args.column_wise)

        L, N, F = data.shape

        feature_list = [data]

        # numerical time_in_day
        time_ind = [i % args.steps_per_day / args.steps_per_day for i in range(data.shape[0])]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

        # numerical day_in_week
        day_in_week = [(i // args.steps_per_day) % args.days_per_week for i in range(data.shape[0])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)


    # data = np.concatenate(feature_list, axis=-1)
    x, y = Add_Window_Horizon(data, args.lag, args.horizon, single)
    x_day, y_day = Add_Window_Horizon(time_in_day, args.lag, args.horizon, single)
    x_week, y_week = Add_Window_Horizon(day_in_week, args.lag, args.horizon, single)
    x, y = np.concatenate([x, x_day, x_week], axis=-1), np.concatenate([y, y_day, y_week], axis=-1)

    # spilit dataset by days or by ratio
    if args.test_ratio > 1:
        x_train, x_val, x_test = split_data_by_days(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_days(y, args.val_ratio, args.test_ratio)
        # day_train, day_val, day_test = split_data_by_days(time_in_day, args.val_ratio, args.test_ratio)
        # week_train, week_val, week_test = split_data_by_days(day_in_week, args.val_ratio, args.test_ratio)
    else:
        x_train, x_val, x_test = split_data_by_ratio(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_ratio(y, args.val_ratio, args.test_ratio)
        # week_train, week_val, week_test = split_data_by_ratio(day_in_week, args.val_ratio, args.test_ratio)
    # #normalize st data
    scaler = normalize_dataset(x_train[..., :args.input_dim], normalizer, args.column_wise)
    # # scaler1 = normalize_dataset(data_val, normalizer, args.column_wise)
    # # scaler2 = normalize_dataset(data_test, normalizer, args.column_wise)
    # # scaler3 = normalize_dataset(data, normalizer, args.column_wise)
    #
    x_train[..., :args.input_dim] = scaler.transform(x_train[..., :args.input_dim])
    x_val[..., :args.input_dim] = scaler.transform(x_val[..., :args.input_dim])
    x_test[..., :args.input_dim] = scaler.transform(x_test[..., :args.input_dim])
    # y_train[...,:args.input_dim] = scaler.transform(y_train[...,:args.input_dim])
    # y_val[...,:args.input_dim] = scaler.transform(y_val[...,:args.input_dim])
    # y_test[...,:args.input_dim] = scaler.transform(y_test[...,:args.input_dim])
    # data_train = np.concatenate([data_train,day_train,week_train], axis=-1)
    # data_val = np.concatenate([data_val, day_val, week_val], axis=-1)
    # data_test = np.concatenate([data_test, day_test, week_test], axis=-1)
    # data_len = len(x_train)
    # indices = list(range(data_len))
    data_len = N
    indices = list(range(N))

    # 计算每个客户端的数据大小
    partition_method = getattr(args, 'client_partition_method', 'random')
    if partition_method == 'metis_dtw':
        client_indices = split_node_indices_by_dtw_metis(data, args, num_clients)
    elif partition_method == 'random':
        random.shuffle(indices)
        client_indices = split_node_indices(indices, num_clients, getattr(args, 'client_node_ratio', None))
    else:
        raise ValueError('client_partition_method must be random or metis_dtw')

    client_loaders = []
    client_node_id = []

    for i, subset_indices in enumerate(client_indices):
        # 如果当前是余数的部分，则该客户端数据多一个
        # 截取属于第 i 个用户的数据索引
        subset_trainx = x_train[:, :, subset_indices, :]
        subset_trainy = y_train[:, :, subset_indices, :]
        subset_valx = x_val[:, :, subset_indices, :]
        subset_valy = y_val[:, :, subset_indices, :]
        subset_testx = x_test[:, :, subset_indices, :]
        subset_testy = y_test[:, :, subset_indices, :]

        # 将每个客户端的数据保存到列表
        # client_loaders.append((subset_trainx, subset_trainy, subset_valx, subset_valy, subset_testx, subset_testy))
        # client_node_id.append(subset_indices)
        # 封装成 DataLoader
        train_loader = data_loader(subset_trainx, subset_trainy, args.batch_size, shuffle=True, drop_last=True)
        val_loader = data_loader(subset_valx, subset_valy, args.batch_size, shuffle=False, drop_last=True)
        test_loader = data_loader(subset_testx, subset_testy, args.batch_size, shuffle=False, drop_last=False)
            # DataLoader(subset, batch_size=128, shuffle=True)
        loader=[train_loader,val_loader,test_loader]
        client_loaders.append(loader)
        client_node_id.append(subset_indices)
    #     client_datax.append(subsetx)
    #     client_datay.append(subsety)
    # data_test = []
    # for i in range(len(client_datax)):
    #     data_test.append(client_datax[i][:,:,:,0])
    #     print(f"Client {i} feature mean:", np.mean(data))



    # for i in range(len(data_test)):
    #     for j in range(i+1,len(data_test)):
    #         for n in range(N):
    #             data_i = flatten_data(data_test[i][..., n])
    #             data_j = flatten_data(data_test[j][..., n])
    #             ks_stat, p_val = stats.ks_2samp(data_i, data_j)
    #             if p_val < 0.05:
    #                 print(f"Clients {i},{j} differ on dimension {n}: KS {ks_stat:.3f}, p={p_val:.4f}")
                # statistic, pvalue = stats.ks_2samp(data_test[i], data_test[j])
                # print(f"Client {i} and {j} pvalue: is",pvalue)
    # #add time window
    # x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    # x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    # x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_train.shape, y_train.shape)
    print('Client_Node_Counts: ', [len(i) for i in client_node_id])
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    ##############get dataloader######################
    # train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True, drop_last=True)
    # if len(x_val[..., 0]) == 0:
    #     val_dataloader = None
    # else:
    #     val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    # test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    # return client_loaders, client_node_id, train_dataloader, val_dataloader, test_dataloader, scaler
    return client_loaders, client_node_id, None, None, None, scaler

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)
