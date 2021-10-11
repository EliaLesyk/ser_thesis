import argparse
import torch as t
from matplotlib import pyplot as plt
import numpy as np
#from sklearn.model_selection import train_test_split
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import get_device
from prepare_data import load_data
from preprocess_audio import NUMCEP as numcep, MFCC_DUR as mfcc_dur, SPECTR_DUR as spectr_dur
from models import CNN_MFCC, CNN_Spectr, Conv_GRU, AttentionLSTM
from trainer import Trainer

# some hyper parameters
#BATCH_SIZE = 8
#EPOCHS = 10
LEARN_RATE = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default="conv_gru")
    parser.add_argument("batch_size", type=int, default=64)
    parser.add_argument("epochs", type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    print("Used device is:", device)

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # create an instance of our model
    if args.model_name == "conv_gru":
        unsqueeze_needed = True
        input = "mfcc"
        model = Conv_GRU(input_shape=[BATCH_SIZE, 1, numcep, mfcc_dur], hidden_size=32).to(device)
    elif args.model_name == "cnn-mfcc":
        unsqueeze_needed = True
        input = "mfcc"
        model = CNN_MFCC(num_classes=5, kernel_size1=(3,1), kernel_size2=(2,1), dur=mfcc_dur).to(device)
    elif args.model_name == "cnn-spectr":
        unsqueeze_needed = False
        input = "spectr"
        model = CNN_Spectr(num_classes=5, input_size=spectr_dur, pool_size=2, num_filters=[8, 8, 8, 8], \
                           dropout=0.5, conv_size=3).to(device)
    elif args.model_name == "att_lstm":
        unsqueeze_needed = False
        input = "mfcc"
        model = AttentionLSTM(BATCH_SIZE, num_classes=5, hidden_dim=100, emb_dim=50)
    else:
        raise Exception("model_name parameter has to be one of [conv_gru|cnn-mfcc|cnn-spectr|att_lstm] | batch_size | number of epochs")
    
    train_dataset, test_dataset = load_data(input=input)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    criterion = nn.CrossEntropyLoss()
    #criterion = t.nn.BCEWithLogitsLoss()

    # set up the optimizer
    optimizer = Adam(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY) # weight_decay = 0.0001
    #optimizer = t.optim.SGD(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(model, args.model_name, criterion, optimizer, train_loader, test_loader, \
                     cuda=t.cuda.is_available(), early_stopping_patience=20, unsqueeze_needed=unsqueeze_needed)

    #if len(sys.argv) >= 2:
     #   trainer.restore_checkpoint(int(sys.argv[1]))

    # go, go, go... call fit on trainer
    res = trainer.fit(EPOCHS)

    # plot the results
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('losses.png')

