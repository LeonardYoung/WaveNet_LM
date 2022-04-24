import torch.optim as optim
from model import *
import utils.util as util
import config as Config
import stgcn as STGCN
import torch

class trainer():
    def __init__(self, scaler , device, supports):

        nhid = 32
        if Config.model_name == 'waveNet':
            self.model = gwnet(device,Config.dropout, supports=supports,
                               in_dim=Config.in_dim, out_dim=Config.seq_length, residual_channels=nhid,
                               dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        elif Config.model_name == 'stgcn':
            sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj('data/water/A/adjs/adj.pkl', 'doubletransition')
            self.adj_hat = torch.from_numpy(adj_mx[0]).to(device)

            self.model = STGCN.STGCN(Config.num_nodes,2,24,3)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.device = device

    def input_model(self,input):
        if Config.model_name == 'stgcn':
            input = input.transpose(1,2)
            input = input.transpose(2,3)
            output = self.model(self.adj_hat, input)
            output = output.transpose(1,2)
            output = output.unsqueeze(3)
        else:
            output = self.model(input)
        return output

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        if len(real_val.shape) == 3:
            input = nn.functional.pad(input,(1,0,0,0))
            real = torch.unsqueeze(real_val, dim=1)
        else:
            real = real_val

        output = self.input_model(input)
        output = output.transpose(1,3)

        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        if len(real_val.shape) == 3:
            input = nn.functional.pad(input,(1,0,0,0))
            real = torch.unsqueeze(real_val, dim=1)
        else:
            real = real_val
        output = self.input_model(input)
        output = output.transpose(1,3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
