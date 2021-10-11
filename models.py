import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

# CNN Network
class CNN_MFCC(nn.Module):
    def __init__(self, num_classes=5, kernel_size1=(3,1), kernel_size2=(2,1), dur=50):
        super(CNN_MFCC, self).__init__()

        # Input shape = (batch_size, 1, numceps=40, mfcc_dur=50)
        # Input (batch_size, num_channels, height, width)
        # Output size after convolution filter = ((w-kernel_size/filter_size+2P)/s) +1
        self.dur = dur
        self.num_classes = num_classes

        #kernel_size=(3,1) not to influence time dimension
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size1, stride=1)
        # Shape = ([batch_size, 8, (40-3+1)/1, 50])
        self.bn1 = nn.BatchNorm2d(num_features=8) # num_features = num_filters
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size2)     # (2,1) or 2
        # Reduce the image size be factor 2
        # Shape = (batch_size,8,19,50)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size1, stride=1)
        # Shape = (batch_size, 16, 17, 50)
        self.relu2 = nn.ReLU()

        # TODO: either expanding or narrowing

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=5, kernel_size=kernel_size2, stride=1)
        # Shape = (batch_size, 5, 16, 50)
        self.bn3 = nn.BatchNorm2d(num_features=5)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(in_features=self.num_classes*16*self.dur, out_features=num_classes)

    # Feed forward function
    def forward(self, input):
        output = self.conv1(input)
        #print("output shape after conv1", output.shape)
        output = self.bn1(output)
        #print("output shape after bn1", output.shape)
        output = self.relu1(output)
        #print("output shape after relu1", output.shape)

        output = self.pool(output)
        #print("output shape after pool", output.shape)

        output = self.conv2(output)
        #print("output shape after conv2", output.shape)
        output = self.relu2(output)
        #print("output shape after relu2", output.shape)

        output = self.conv3(output)
        #print("output shape after conv3", output.shape)
        output = self.bn3(output)
        #print("output shape after bn3", output.shape)
        output = self.relu3(output)
        #print("output shape after relu3", output.shape)

        #output = output.view(-1, 5*16*self.dur)
        output = output.view(-1, self.num_classes * 16 * self.dur)
        output = self.fc(output)
        return output


class CNN_Spectr(nn.Module):
    def __init__(self, num_classes=5, input_size=128, pool_size=2, num_filters=[8, 8, 8, 8], dropout=0.5, conv_size=3):
        super(CNN_Spectr, self).__init__()
        self.pool_size = pool_size
        self.conv_size = conv_size
        self.conv_layers = self._build_conv_layers(num_filters)
        # output size after convolution = ((width - kernel_size + 2*padding)/stride + 1
        self.out_size = input_size / (pool_size**len(num_filters))
        #self.flat_size = cfg.num_filters[len(cfg.num_filters)-1] * self.out_size**2
        #self.flat_size = int(cfg.num_filters[len(cfg.num_filters) - 1] * self.out_size ** 2)
        self.flat_size = int(num_filters[len(num_filters) - 1] * self.out_size **2)
        self.fc2 = nn.Linear(self.flat_size, num_classes)
        # torch.nn.Linear(in_features=depth*height*width, out_features=num_classes)
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


class Conv_GRU(nn.Module):

    def __init__(self, input_shape, hidden_size, bidirectional=True, nlayers_rnn=2, n_classes=5):
        super(Conv_GRU, self).__init__()

        self.input_shape = input_shape  # [batch size, channels, sequence length, # MFCCs]
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.nlayers_Bigru = nlayers_rnn
        self.n_classes = n_classes
        self.n_ch = 4
        self.kernel = (3, 1)  # The kernel is only applied to the mfccs dimension

        self.Conv2d = t.nn.Conv2d(self.input_shape[1], self.n_ch, self.kernel)
        self.BatchNorm_conv = t.nn.BatchNorm2d(self.n_ch)  # Output

        outConvShape = self._get_conv_output(self.input_shape)
        # Number of features
        input_size = self.n_ch * outConvShape  # input_size=num_channels*num_dimensions

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
        output_biGru = int((self.hidden_size * idx_bi)) * self.input_shape[3]
        self.BatchNorm_biGru = t.nn.BatchNorm1d(self.input_shape[3])  # Output
        # out of GRU: hidden_size*N_time_frames*idx_bi

        self.fc = t.nn.Linear(output_biGru, self.n_classes)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        # generates a variable equal to the input to compute automatically the dimensions for the output of the conv and the input of the gru
        inputd = Variable(t.rand(*shape))
        output_feat = self._forward_Conv(inputd)
        # As the sequence is static for the gru and the dimension that we transform was mfccs we extract the output of the dimension 3
        n_size = output_feat.size(2)  # output_feat.data.view(inputd.size()[0], -1).size(1)
        return n_size

    def _forward_Conv(self, x):
        """
        Convolutional layer features
        ReLU, and max pooling
        """

        out = self.Conv2d(x)
        out = F.max_pool2d(self.BatchNorm_conv(out), (2, 1))
        out = F.elu(out)

        return out

    def forward(self, input_tensor):
        """
        GRU layer
        btachnorm and droupout
        """

        out = self._forward_Conv(input_tensor)

        # GRU
        out = out.permute(0, 3, 1, 2)  # Permute dimensions to keep one-to-one context
        # from [batch size, channels, sequence length, # MFCCs] to [batch size, sequence length, channels, # MFCCs]
        out = out.contiguous().view(out.shape[0], out.shape[1], -1)
        # Concatenate sequence length and # MFCCs resulting
        # [batch size, sequence length, channels*# MFCCs]
        out, _ = self.GRU(out)
        print("out shape before batchnorm", out.shape)
        out = self.BatchNorm_biGru(out)
        out = F.dropout(F.elu(out))

        out = self.fc(out.view(out.size(0), -1))

        return out


class AttentionLSTM(nn.Module):
    """Taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py"""
    def __init__(self, batch_size, num_classes=5, hidden_dim=100, emb_dim=50):
        """
        LSTM with self-Attention model.
        """
        super(AttentionLSTM, self).__init__()
        self.batch_size = batch_size
        self.output_size = num_classes
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
        attn_output = self.extract(input)
        logits = self.classify(attn_output)
        return logits
