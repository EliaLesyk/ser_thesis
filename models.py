import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import math


class CNN(nn.Module):
    def __init__(self, n_classes=5, kernel1=(3,1), kernel2=(2,1), dur=50, numcep=40, input_type="mfcc"):
        super(CNN, self).__init__()

        # Input shape = (batch_size, 1, numceps=40, mfcc_dur=50)
        # Input (batch_size, num_channels, height, width)
        # Output size after convolution filter = ((w-kernel_size/filter_size+2P)/s) +1
        self.dur = dur
        self.numcep = numcep
        self.n_classes = n_classes
        self.input_type = input_type
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        #kernel_size=(3,1) not to influence time dimension
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=kernel1) # stride=1
        self.bn1 = nn.BatchNorm2d(num_features=64)      # num_features = num_filters
        self.relu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=kernel2)     # (2,1) or 2
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=kernel2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ELU()
        #self.pool3 = nn.MaxPool2d(kernel_size=kernel_size2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=5, kernel_size=kernel1)
        self.bn4 = nn.BatchNorm2d(num_features=self.n_classes)
        self.relu4 = nn.ELU()
        #self.pool4 = nn.MaxPool2d(kernel_size=kernel_size2)
        self.drop4 = nn.Dropout(0.25)

        out_lin_dim = self._calc_lin_dim(self.numcep, self.dur, self.kernel1, self.kernel2)
        self.fc = nn.Linear(in_features=out_lin_dim, out_features=256)
        self.relu4 = nn.GELU()
        self.fc1 = nn.Linear(in_features=256, out_features=self.n_classes)

    def _calc_lin_dim(self, numcep, dur, kernel1, kernel2):
        # (Hout, Wout) conv with no stride & pool with stride = kernel by default
        out1 = (int(math.ceil((numcep - (kernel1[0]-1) - (kernel2[0]-1))/kernel2[0])),
                int(math.ceil((dur - (kernel1[1]-1) - (kernel2[1]-1)) / kernel2[1])))
        out2 = (int((math.ceil(out1[0] - (kernel1[0]-1) - (kernel2[0]-1)) / kernel2[0])),
                (int(math.ceil((out1[1] - (kernel1[1]-1) - (kernel2[1]-1)) / kernel2[1]))))

        # conv only
        out3 = (int(math.ceil(out2[0] - (kernel1[0]-1))), math.ceil(int(out2[1] - (kernel1[1]-1))))
        out4 = (int(math.ceil(out3[0] - (kernel1[0]-1))), int(math.ceil(out3[1] - (kernel1[1]-1))))
        out_lin = out4[0] * out4[1] * self.n_classes      # 5 because of batch_norm bn4 num_features
        return out_lin

    # Feed forward function
    def forward(self, input):
        #print("input shape", input.shape)
        output = self.conv1(input)
        #print("output shape after conv1", output.shape)     # mspectr: ([batch_size, 8, 126, 50])/(bs, 8, 38, 50)
        output = self.bn1(output)
        #print("output shape after bn1", output.shape)       # mspectr: ([bs, 8, 126, 50]) /mfcc:([bs, 8, 48, 40])
        output = self.relu1(output)
        # print("output shape after relu1", output.shape)
        output = self.pool1(output)
        #print("output shape after pool1", output.shape)      # mspectr (bs, 8, 63, 50) /mfcc
        output = self.drop1(output)

        output = self.conv2(output)
        #print("output shape after conv2", output.shape)     # mspectr (bs, 16, 61, 50) /mfcc (bs, 16, 17, 50)
        output = self.bn2(output)
        output = self.relu2(output)
        #print("output shape after relu2", output.shape)
        output = self.pool2(output)
        #print("output shape after pool2", output.shape)      # mspectr (bs, 8, 63, 50) /mfcc
        output = self.drop2(output)
        #print("output shape after drop2", output.shape)

        output = self.conv3(output)
        #print("output shape after conv3", output.shape)     # mspectr (bs, 5, 60, 50) /mfcc (bs, 5, 16, 50)
        output = self.bn3(output)
        #print("output shape after bn3", output.shape)
        output = self.relu3(output)
        #print("output shape after relu3", output.shape)     # mspectr ([bs, 5, 60, 50])/ mfcc (bs, 5, 16, 50)
        #output = self.pool3(output)             # pooling layer
        output = self.drop3(output)

        output = self.conv4(output)
        #print("output shape after conv4", output.shape)     # mspectr (bs, 5, 60, 50) /mfcc (bs, 5, 16, 50)
        output = self.bn4(output)
        #print("output shape after bn4", output.shape)
        output = self.relu4(output)
        #print("output shape after relu4", output.shape)     # mspectr ([bs, 5, 60, 50])/ mfcc (bs, 5, 16, 50)
        #output = self.pool4(output)             # pooling layer
        output = self.drop4(output)
        #print("output shape after drop4", output.shape)

        #output = output.view(-1, 5 * 60 * self.dur)
        output = output.view(output.size(0), -1)        # classification layer
        #print("after view", output.shape)

        output = self.fc(output)
        #print("after fc", output.shape)
        output = self.relu4(output)             # GELU
        output = self.fc1(output)
        #print("after fc1", output)

        return output


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, out_dim=4):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = out_dim

        ## 1st hidden layer
        self.linear_1 = nn.Linear(self.in_dim, self.hidden_dim_1)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        self.linear_1_bn = nn.BatchNorm1d(self.hidden_dim_1, momentum=0.6)

        ## 2nd hidden layer
        self.linear_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()
        self.linear_2_bn = nn.BatchNorm1d(self.hidden_dim_2, momentum=0.6)

        ## Out layer
        self.linear_out = nn.Linear(self.hidden_dim_2, self.out_dim)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):

        out = self.linear_1(x)
        #print(out.size())
        out = self.linear_1_bn(out)
        out = F.relu(out)


        out = self.linear_2(out)
        out = self.linear_2_bn(out)  # last layer
        # print(out.size())

        out = F.relu(out)
        out = F.dropout(out, p=0.175, training=self.training)
        #print(out.size())

        out = self.linear_out(out)
        #print(out.size())
        return out


class CNN_Spectr(nn.Module):
    def __init__(self, n_classes=5, input_size=128, pool_size=2, num_filters=[8, 8, 8, 8], dropout=0.5, conv_size=3):
        super(CNN_Spectr, self).__init__()
        self.pool_size = pool_size
        self.conv_size = conv_size
        self.conv_layers = self._build_conv_layers(num_filters)
        # output size after convolution = ((width - kernel_size + 2*padding)/stride + 1
        self.out_size = input_size / (pool_size**len(num_filters))
        #self.flat_size = cfg.num_filters[len(cfg.num_filters)-1] * self.out_size**2
        #self.flat_size = int(cfg.num_filters[len(cfg.num_filters) - 1] * self.out_size ** 2)
        self.flat_size = int(num_filters[len(num_filters) - 1] * self.out_size **2)
        self.fc2 = nn.Linear(self.flat_size, n_classes)
        # torch.nn.Linear(in_features=depth*height*width, out_features=n_classes)
        self.dropout = nn.Dropout(dropout)

    def _build_conv_layers(self, num_filters):
        conv_layers = []
        num_channels = [1] + num_filters
        for i in range(len(num_channels)-1):
            conv_layers.append(nn.Conv2d(num_channels[i], num_channels[i+1], self.conv_size, padding=1))
            # nn.Conv2d(n_channels, out_channels, kernel_size, stride = 1, padding = 0)
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(self.pool_size, self.pool_size)) # width/pool_size and height/pool_size
        return nn.Sequential(*conv_layers)

    def extract(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_size)
        return x

    def classify(self, x):
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.classify(x)
        return x

"""
class Conv_GRU(nn.Module):

    def __init__(self, input_shape, hidden_size, bidirectional=True, nlayers_rnn=2, n_classes=5):
        super(Conv_GRU, self).__init__()

        self.input_shape = input_shape  # [batch size, channels, sequence length, # MFCCs]
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.nlayers_Bigru = nlayers_rnn
        self.n_classes = n_classes
        self.n_ch = 4
        self.kernel = 3
        self.kernel1 = 2

        self.Conv = t.nn.Conv2d(self.input_shape[1], 64, self.kernel)
        self.BatchNorm_conv = t.nn.BatchNorm2d(64)
        self.drop = t.nn.Dropout(0.5)

        self.Conv1 = t.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel1)
        self.Conv2 = t.nn.Conv2d(in_channels=16, out_channels=self.n_classes, kernel_size=self.kernel)
        self.drop1 = t.nn.Dropout(0.25)
        self.BatchNorm_conv1 = t.nn.BatchNorm2d(num_features=self.n_classes)

        outConvShape = self._get_conv_output(self.input_shape)

        # Number of features
        #input_size = self.n_ch * outConvShape  # input_size=num_channels*num_dimensions
        #print("input", input_size)
        input_size = 20
        self.GRU = t.nn.GRU(input_size, self.hidden_size,
                                bidirectional=self.bidirectional,
                                num_layers=self.nlayers_Bigru,
                                batch_first=True)

        # In case of bidirectional recurrent network
        idx_bi = 1
        if self.bidirectional:
            idx_bi = 2
        # Linear transformation - Affine mapping
        # As we are concatenating everything for the linear layer->
        output_biGru = int((self.hidden_size * idx_bi)) * self.input_shape[3]   #32*2*50

        self.BatchNorm_biGru = t.nn.BatchNorm1d(self.input_shape[3])  # Output
        # out of GRU: hidden_size*N_time_frames*idx_bi

        self.drop4 = t.nn.Dropout(0.25)
        self.fc = t.nn.Linear(output_biGru, 256)
        self.fc1 = t.nn.Linear(in_features=256, out_features=self.n_classes)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        # generates a variable equal to the input to compute automatically the dimensions for the output of the conv and the input of the gru
        inputd = Variable(t.rand(*shape))
        output_feat = self._forward_Conv(inputd)
        # As the sequence is static for the gru and the dimension that we transform was mfccs we extract the output of the dimension 3
        n_size = output_feat.size(2)
        #n_size = output_feat.data.view(inputd.size()[0], -1).size(1)
        return n_size

    def _forward_Conv(self, x):
        
        #Convolutional layer features
        #ReLU, and max pooling


        out = self.Conv(x)
        out = F.elu(self.BatchNorm_conv(out))
        out = F.max_pool2d(out, kernel_size=3)
        out = nn.drop(out)

        out = self.Conv1(out)
        out = F.elu(self.BatchNorm_conv(out))
        out = F.max_pool2d(out, kernel_size=4)
        out = nn.drop1(out)

        out = self.Conv1(out)
        out = F.max_pool2d(self.BatchNorm_conv1(out), (2,1))
        out = F.elu(out)

        out = self.Conv2(out)
        out = F.max_pool2d(self.BatchNorm_conv1(out), (2, 1))
        out = F.elu(out)

        return out

    def forward(self, input_tensor):

        #GRU layer
        #batchnorm and dropout

        out = self._forward_Conv(input_tensor)

        # GRU
        out = out.permute(0, 3, 1, 2)  # Permute dimensions to keep one-to-one context
        # from [batch size, channels, sequence length, # MFCCs] to [batch size, sequence length, channels, # MFCCs]
        out = out.contiguous().view(out.shape[0], out.shape[1], -1)
        # Concatenate sequence length and # MFCCs resulting
        # [batch size, sequence length, channels*# MFCCs]
        #print("out shape before GRU", out.shape)  # mfcc torch.Size([8, 50, 76]) # mspectr torch.Size([8, 50, 252])
        out, _ = self.GRU(out)

        out = self.BatchNorm_biGru(out)

        out = self.fc(out.view(out.size(0), -1))
        out = self.drop(F.gelu(out))
        #out = F.gelu(out)

        #out = self.fc(out)
        out = self.fc1(out)     # added

        return out
"""


class Conv_GRU(nn.Module):

    def __init__(self, input_shape, hidden_size, kernel1, kernel2, bidirectional=True, nlayers_rnn=2, n_classes=5):
        super(Conv_GRU, self).__init__()

        self.input_shape = input_shape  # [batch size, channels, sequence length, # MFCCs]
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.nlayers_Bigru = nlayers_rnn
        self.n_classes = n_classes
        self.n_ch = 4
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        #CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=self.kernel1)  # stride=1
        self.bn1 = nn.BatchNorm2d(num_features=64)  # num_features = num_filters
        self.relu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel2)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=self.kernel2)     # stride=(4,4)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=self.kernel2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=self.n_classes, kernel_size=self.kernel1)
        self.bn4 = nn.BatchNorm2d(num_features=self.n_classes)
        self.relu4 = nn.ELU()
        self.pool4 = nn.MaxPool2d(kernel_size=self.kernel2)
        self.drop4 = nn.Dropout(0.25)

        outConvShape = self.get_conv_output(self.input_shape)
        # Number of features
        #input_size = self.n_ch * outConvShape   # 4*140

        # GRU
        self.GRU = t.nn.GRU(outConvShape[2], self.hidden_size,
                                bidirectional=self.bidirectional,
                                num_layers=self.nlayers_Bigru,
                                batch_first=True)

        self.BatchNorm_biGru = t.nn.BatchNorm1d(outConvShape[1])  # Output

        # In case of bidirectional recurrent network
        idx_bi = 1
        if self.bidirectional:
            idx_bi = 2

        # Linear transformation - Affine mapping
        # As we are concatenating everything for the linear layer->
        #output_biGru = int((self.hidden_size * idx_bi)) * self.input_shape[3]
        output_biGru = int(self.hidden_size * idx_bi * outConvShape[1])     # hid_size * bidir_index * seq_len
        self.drop_gru = t.nn.Dropout(0.25)
        self.fc_gru1 = t.nn.Linear(output_biGru, 256)
        self.fc_gru2 = t.nn.Linear(in_features=256, out_features=self.n_classes)

    # generate input sample and forward to get shape
    def get_conv_output(self, shape):
        # generates a variable equal to the input to compute automatically the dimensions for the output of the conv and the input of the gru
        #input = Variable(t.rand(*shape))       # * to switch from 0.abcd to scientific notation
        input = t.rand(shape)
        output = self._forward_Conv(input)
        output = output.permute(0, 3, 1, 2)
        out = output.contiguous().view(output.shape[0], output.shape[1], -1)
        return out.shape

    def _forward_Conv(self, input):
        """
        Convolutional layer features
        ReLU, and max pooling
        """
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool1(output)
        output = self.drop1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        output = self.drop2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.drop3(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.drop4(output)

        return output

    def forward(self, input_tensor):
        """
        GRU layer
        batchnorm and dropout
        """

        out = self._forward_Conv(input_tensor)
        out = out.permute(0, 3, 1, 2)  # Permute dimensions to keep one-to-one context
        # from [batch size, channels, sequence length, # MFCCs] to [batch size, sequence length, channels, # MFCCs]
        out = out.contiguous().view(out.shape[0], out.shape[1], -1)
        # Concatenate sequence length and resulting # MFCCs -> [batch size, sequence length, channels*# MFCCs]

        # GRU
        out, _ = self.GRU(out)
        out = self.BatchNorm_biGru(out)
        out = self.fc_gru1(out.view(out.size(0), -1))
        out = self.drop_gru(F.gelu(out))
        out = self.fc_gru2(out)

        return out


class AttentionLSTM(nn.Module):
    """Adapted from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py"""
    def __init__(self, batch_size, n_classes=5, hidden_dim=100, emb_dim=50):
        """
        LSTM with self-Attention model.
        """
        super(AttentionLSTM, self).__init__()
        self.batch_size = batch_size
        self.output_size = n_classes
        self.hidden_size = hidden_dim
        self.embedding_length = emb_dim     # length of sequence

        self.dropout = nn.Dropout(p=0.0)
        self.dropout2 = nn.Dropout(p=0.0)

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, final_state):
        """
        This method computes soft alignment scores for each of the hidden_states and the last hidden_state of the LSTM.
        Tensor Sizes :
            hidden.shape = (batch_size, hidden_size)
            attn_weights.shape = (batch_size, num_seq)
            soft_attn_weights.shape = (batch_size, num_seq)
            new_hidden_state.shape = (batch_size, hidden_size)

        :param lstm_output: Final output of the LSTM which contains hidden layer outputs for each sequence.
        :param final_state: Final time-step hidden state (h_n) of the LSTM
        :return: Context vector produced by performing weighted sum of all hidden states with attention weights
        """

        hidden = final_state.squeeze(0)
        attn_weights = t.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = t.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # t.bmm - batch matrix-matrix product of matrices
        return new_hidden_state

    def extract(self, input):
        input = input.transpose(0, 1)
        input = self.dropout(input)

        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2)

        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def classify(self, attn_output):
        attn_output = self.dropout2(attn_output)
        logits = self.label(attn_output)
        return logits.squeeze(1)

    def forward(self, input):
        #print("input shape", input.shape)
        attn_output = self.extract(input)
        logits = self.classify(attn_output)
        return logits

"""
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        
        #Initialize the ConvLSTM cell
        #:param input_size: (int, int)
         #   Height and width of input tensor as (height, width).
        #:param input_dim: int
         #   Number of channels of input tensor.
        #:param hidden_dim: int
         #   Number of channels of hidden state.
        #:param kernel_size: (int, int)
         #   Size of the convolutional kernel.
        #:param bias: bool
         #   Whether or not to add the bias.
        #:param dtype: torch.cuda.FloatTensor or torch.FloatTensor
         #   Whether or not to use cuda.
        
        super(ConvGRUCell, self).__init__()
        self.height = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return Variable(t.zeros(batch_size, self.hidden_dim, self.height)).type(self.dtype)

    def forward(self, input_tensor, h_cur):
        #:param self:
        #:param input_tensor: (b, c, h, w)
         #   input is actually the target_model
        #:param h_cur: (b, c_hidden, h, w)
         #   current hidden and cell states respectively
        #:return: h_next,
         #   next hidden state
        combined = t.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = t.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = t.sigmoid(gamma)
        update_gate = t.sigmoid(beta)

        combined = t.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = t.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU_Test(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        
        #:param input_size: (int, int)
         #   Height and width of input tensor as (height, width).
        #:param input_dim: int e.g. 256
         #   Number of channels of input tensor.
        #:param hidden_dim: int e.g. 1024
         #   Number of channels of hidden state.
        #:param kernel_size: (int, int)
         #   Size of the convolutional kernel.
        #:param num_layers: int
         #   Number of ConvLSTM layers
        #:param dtype: torch.cuda.FloatTensor or torch.FloatTensor
         #   Whether or not to use cuda.
        #:param alexnet_path: str
         #   pretrained alexnet parameters
        #:param batch_first: bool
         #   if the first position of array is batch or not
        #:param bias: bool
         #   Whether or not to add the bias.
        #:param return_all_layers: bool
         #   if return hidden and cell states for all layers
        
        super(ConvGRU_Test, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        #self.height, self.width = input_size
        self.height = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(#input_size=(self.height, self.width),
                                         input_size=self.height,
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        
        #param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
         #   extracted features from alexnet
        #param hidden_state:
        #return: layer_output_list, last_state_list
        
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            #input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            input_tensor = input_tensor.permute(1, 0, 2)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            print("h shape", h.shape)
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                #h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)
                                              # h_cur=h)
                h  = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :], # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = t.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class EncoderLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0):
    super(EncoderLSTM, self).__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

  def forward(self, inputs, hidden):
    # Embed input words
    embedded = self.embedding(inputs)
    # Pass the embedded word vectors into LSTM and return all outputs
    output, hidden = self.lstm(embedded, hidden)
    return output, hidden

  def init_hidden(self, batch_size=1):
    return (t.zeros(self.n_layers, batch_size, self.hidden_size),
            t.zeros(self.n_layers, batch_size, self.hidden_size))


class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(t.FloatTensor(1, hidden_size))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()
        # Embed input words
        embedded = self.embedding(inputs).view(1, -1)
        embedded = self.dropout(embedded)

        # Calculating Alignment Scores
        x = t.tanh(self.fc_hidden(hidden[0]) + self.fc_encoder(encoder_outputs))
        alignment_scores = x.bmm(self.weight.unsqueeze(2))

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = t.bmm(attn_weights.unsqueeze(0),
                                   encoder_outputs.unsqueeze(0))

        # Concatenating context vector with embedded input word
        output = t.cat((embedded, context_vector[0]), 1).unsqueeze(0)
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)
        # Passing the LSTM output through a Linear layer acting as a classifier
        output = F.log_softmax(self.classifier(output[0]), dim=1)
        return output, hidden, attn_weights


class LuongDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=1, drop_prob=0.1):
        super(LuongDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # The Attention Mechanism is defined in a separate class
        self.attention = attention

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embed input words
        embedded = self.embedding(inputs).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(lstm_out, encoder_outputs)
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = t.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        # Concatenating output from LSTM with context vector
        output = t.cat((lstm_out, context_vector), -1)
        # Pass concatenated vector through Linear layer acting as a Classifier
        output = F.log_softmax(self.classifier(output[0]), dim=1)
        return output, hidden, attn_weights


class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(t.FloatTensor(1, hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = t.tanh(self.fc(decoder_hidden + encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)
"""
