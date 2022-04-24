
model_name = 'waveNet'
# model_name = 'stgcn'
# section = 'B'
section = 'A'
seed = 42
epoch = 2000
patience = 50
batch_size = 64
in_dim = 2
device = 'cuda:1'
dropout = 0.3
learning_rate = 0.001
weight_decay = 0.0001    # weight decay rate
seq_length = 3 # output length
aptonly = True
data_dir = f'data'
adjdata = f'data/adj_all_one.pkl'
num_factors = 6     # number of factor
num_nodes = 10      # number of nodes(sites)
num_site = 10  # number of nodes(sites)
input_data_len = 24 # input seq length

gcn_bool = True             # whether to add graph convolution layer
adj_learn_type = 'weigthed'       # adj learn method
use_LSTM = True

fac_single = True # single factor
fac_index = 1 # factor index


# save dir
out_dir = 'tempModel'
# out_dir = 'GCNLSTM'
# out_dir = 'GCNnoLSTM'
# out_dir = 'noGCNLSTM'
# out_dir = 'noGCNnoLSTM'
# out_dir = 'STGCN'

def print_param():
    print(f'adj_learn_type={adj_learn_type}\ngcn_bool={gcn_bool}\nuse_LSTM={use_LSTM}')