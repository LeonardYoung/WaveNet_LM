import pandas as pd
import numpy as np
import os

def merge_one_factor(input_file, inc ,out_dir):
    df = pd.read_csv(input_file, usecols=[1, 2, 3])
    merge = None
    for site in df['sites'].unique():
        one = df.loc[df['sites'] == site]
        one.columns = ['time', 'site'] + [site + str(i) for i in range(1)]
        one = one[['time'] + [site + str(inc)]]
        if merge is None:
            merge = one
        else:
            merge = pd.merge(merge, one, on='time')
    merge.set_index(keys='time', inplace=True)
    exi = os.path.exists(out_dir)
    if not exi:
        os.mkdir(out_dir)

    file_name = out_dir + '/merge' + str(inc)
    merge.to_hdf(file_name+'.h5', key='merge', index=False)
    merge.to_csv(file_name+'.csv')
    return file_name+ '.h5'


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)

    xt = np.arange(0, len(x_offsets), 1) / len(x_offsets)
    xt = np.expand_dims(xt, axis=-1)
    xt = np.tile(xt, [1, num_nodes])
    xt = np.expand_dims(xt, axis=-1)

    yt = np.arange(0, len(y_offsets), 1) / len(y_offsets)
    yt = np.expand_dims(yt, axis=-1)
    yt = np.tile(yt, [1, num_nodes])
    yt = np.expand_dims(yt, axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        one = np.concatenate([data[t + x_offsets, ...], xt], axis=-1)
        x.append(one)
        one = np.concatenate([data[t + y_offsets, ...], yt], axis=-1)
        y.append(one)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y


def generate_dataset(hdf_file, out_dir,seq_length_x=24, seq_length_y=3):
    # seq_length_x, seq_length_y = 24, 24
    df = pd.read_hdf(hdf_file)
    exi = os.path.exists(out_dir)
    if not exi:
        os.mkdir(out_dir)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
    )
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # 随机打乱！
    per = np.random.permutation(x.shape[0])
    x = x[per]
    y = y[per]

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(out_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


# generate a sample adj file
def get_adj_file(sites,file_name):

    id_to_inc = {}
    for i in range(len(sites)):
        id_to_inc[sites[i]] = i

    # generate adj
    ones = np.ones([len(sites), len(sites)])
    adj = ones

    pickle_data = [sites, id_to_inc, adj]
    import pickle
    with open(file_name, "wb") as myprofile:
        pickle.dump(pickle_data, myprofile)


import utils.util as util
if __name__ == "__main__":
    util.set_seed(43)
    fac_idx = 0
    file_name = merge_one_factor('data/sample.csv',fac_idx, 'data/water')
    generate_dataset(file_name, f'data/',24,3)

    # # generate a sample adj file
    # sites = ['site0', 'site1', 'site2', 'site3']
    # get_adj_file(sites,'data/adj_all_one.pkl')

