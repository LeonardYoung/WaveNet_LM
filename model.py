import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import config as Config


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class nconvGLM(nn.Module):
    def __init__(self):
        super(nconvGLM,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)





class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True,num_nodes=10):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        device = Config.device
        self.weightLeft = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device), requires_grad=True).to(device)
        self.weight = nn.Parameter(torch.randn(in_features, out_features).to(device), requires_grad=True).to(device)
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # # AXW
        # support = torch.einsum('ncvl,lo->ncvo', (input, self.weight))
        # output  = torch.einsum('wv,ncvl->ncwl', (adj[0],support))

        # # WAXW
        # adj = torch.mm(self.weightLeft, adj[0])
        # support = torch.einsum('ncvl,lo->ncvo', (input, self.weight))
        # output = torch.einsum('wv,ncvl->ncwl', (adj[0], support))

        # WAX
        adj = torch.mm(self.weightLeft, adj[0])
        output = torch.einsum('wv,ncvl->ncwl', (adj, input))

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len,order,num_nodes,device):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)
        self.weight = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gcnGLM(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len,order,num_nodes,device):
        super(gcnGLM,self).__init__()
        self.nconvGLM = nconvGLM()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)
        self.weight = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)

    def forward(self,x,adjs):
        out = [x]
        x1 = self.nconvGLM(x,adjs)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconvGLM(x1,adjs)
            out.append(x2)
            x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GLM(nn.Module):
    def __init__(self, in_dim,num_node,input_length):
        super(GLM, self).__init__()
        self.in_dim = in_dim
        self.num_node = num_node
        self.input_length = input_length

        self.hidden = 32

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=self.hidden,
                                    kernel_size=(1,1))
        self.convs = nn.ModuleList()

        for i in range(6):
            self.convs.append(nn.Conv2d(in_channels=self.hidden,out_channels=self.hidden,kernel_size=(1,2),dilation=(1,1)))
            self.convs.append(nn.Conv2d(in_channels=self.hidden,out_channels=self.hidden,kernel_size=(1,2),dilation=(1,3)))
        self.end_mlp = nn.Linear(self.hidden,self.num_node)


        # only diag
        self.mlp1 = nn.Linear(input_length,1)
        self.mlp2 = nn.Linear(self.num_node,1)
        self.mlp3 = nn.Linear(self.hidden,self.num_node -1)

    def forward(self, input):

        # mlp
        x = F.relu(self.start_conv(input))
        x = F.relu(self.mlp1(x))
        x = x.squeeze(dim=3)
        # x = x.transpose(1, 2)
        x = F.relu(self.mlp2(x))
        x = x.squeeze(dim=2)
        x = F.relu(self.mlp3(x))
        x = F.dropout(x, 0.3, training=self.training)

        # generate adj
        batch_size = Config.batch_size
        device = Config.device
        x_matrix = torch.zeros([batch_size,self.num_node,self.num_node]).to(device)
        for i in range(self.num_node - 1):
            x_matrix[:,i,i+1] = x[:,i]
        adj = torch.zeros([batch_size,self.num_node,self.num_node],requires_grad=True).to(device)
        adj = adj + x_matrix

        return adj


class gcnWeight(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len,order,num_nodes,device,gcn_site_type=True,factor_masks=[]):
        super(gcnWeight,self).__init__()
        self.nconv = nconv()
        if (not Config.fac_single) and False:
            c_in = (1 + 1 + 1+4)*c_in
        else:
            c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        self.gcn_site_type = gcn_site_type
        self.weight = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device), requires_grad=True).to(device)

    def forward(self,x,support,dtw_matrix=None):

        out = [x]
        wa = self.weight

        for a in support:
            wa = torch.mm(wa, a)

        x1 = self.nconv(x, wa)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, wa)
            out.append(x2)
            x1 = x2
        h = torch.cat(out,dim=1)

        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self,device, dropout=0.3, supports=None,
                 in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,blocks=6,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = Config.gcn_bool
        self.device = device
        num_nodes = Config.num_nodes

        self.adjlearn = Config.adj_learn_type

        self.filter_convs = nn.ModuleList()
        self.lstms = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # GLM
        self.GLM = GLM(in_dim,num_nodes,24).to(device)
        self.GLMadjs = None

        # one dimension
        if in_dim == 2:
            self.blocks = 8
        # multiple dimension
        else:
            self.blocks = 6
            # residual_channels = 64
            # end_channels = 64

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.adj = None

        input_data_len = Config.input_data_len
        input_feature = input_data_len
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool :
            if supports is None:
                self.supports = []
            vec_length = num_nodes
            if self.adjlearn == 'embed':
                vec_length = 16
            else:
                vec_length = num_nodes

            # else:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, vec_length).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(vec_length, num_nodes).to(device), requires_grad=True).to(device)
            self.supports_len +=1



        for b in range(self.blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.lstms.append(nn.LSTM(input_size=32,hidden_size=32,num_layers=2))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    if self.adjlearn == 'GLM':

                        self.gconv.append(gcnGLM(dilation_channels,residual_channels,dropout,
                                          support_len=self.supports_len,order=3,num_nodes=num_nodes,device=device))
                    elif self.adjlearn == 'weigthed' or self.adjlearn == 'weigthedOnly' \
                            or self.adjlearn == 'merge3' or self.adjlearn == 'weigthedDTW':
                        self.gconv.append(gcnWeight(dilation_channels, residual_channels, dropout,
                                                    support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                                    device=device))
                    elif self.adjlearn == 'secondaryGraph':
                        gcn_site_type = True
                        self.gconv.append(gcnWeight(dilation_channels, residual_channels, dropout,
                                                    support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                                    device=device,gcn_site_type=gcn_site_type))
                        gcn_site_type = not gcn_site_type
                    elif self.adjlearn == 'gcnOfficial':
                        self.gconv.append(GraphConvolution( input_feature,input_feature,False))
                        input_feature = input_feature - (2-i)
                    else:
                        self.gconv.append(gcn(dilation_channels, residual_channels, dropout,
                                              support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                              device=device))


        if Config.use_LSTM:
            self.lstm_last = nn.LSTM(input_size=skip_channels,hidden_size=64,num_layers=2)
            self.dnn_last = nn.Linear(64,3)
        else:
            self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)

            self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                        out_channels=out_dim,
                                        kernel_size=(1, 1),
                                        bias=True)

        self.receptive_field = receptive_field


    def compute_adj(self,node1,node2):
        return F.softmax(F.relu(torch.mm(node1, node2)), dim=1)


    def forward(self, input):
        input_numpy = input.to('cpu').numpy()
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.supports is not None:

            if self.adjlearn == 'weigthed' or self.adjlearn == 'embed' or self.adjlearn == 'merge3' \
                    or self.adjlearn == 'gcnOfficial':
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                new_supports = self.supports + [adp]
                self.adj = adp

            # GLM obtain adj matrix
            elif self.adjlearn == 'GLM':
                self.GLMadjs = self.GLM(input)

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)

            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:

                if self.adjlearn == 'GLM':
                    x = self.gconv[i](x, self.GLMadjs)
                else :
                    x = self.gconv[i](x, new_supports)

            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)
        # use LSTM
        if Config.use_LSTM:
            x = skip.squeeze(dim=3)
            x = x.transpose(1,2)
            x = self.lstm_last(x)[0]
            x = self.dnn_last(x)
            x = x.transpose(1,2)
            x = x.unsqueeze(3)
        # use CNN
        else:
            x = F.relu(skip)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)

        return x

