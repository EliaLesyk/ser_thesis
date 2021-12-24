import argparse

import numpy as np
import torch as t
# from sklearn.model_selection import train_test_split
# import sys
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import CNN, CNN_Spectr, Conv_GRU, MLP, AttentionLSTM
from prepare_data import load_data
from preprocess_audio import NUMCEP as icp_numcep, ICP_MFCC_DUR as icp_mfcc_dur, rvd_mspectr_dur, SPECTR_DUR as spectr_dur
from trainer import Trainer
from utils import get_device

# some hyper parameters
#BATCH_SIZE = 8
#EPOCHS = 10
#LEARN_RATE = 0.001     # 0.0001 performs worse
#MOMENTUM = 0.9
#WEIGHT_DECAY = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)     # default="gru"
    parser.add_argument("dataset", type=str)        # default="iemocap"
    parser.add_argument("model_input", type=str)        # default="mspectr"
    parser.add_argument("batch_size", type=int, default=64)
    parser.add_argument("epochs", type=int, default=100)
    parser.add_argument("lr", type=float, default=0.001)
    parser.add_argument("w_decay", type=float, default=0)
    args = parser.parse_args()

    device = get_device()
    print("Used device is:", device)

    if args.dataset == "ravdess":
        dur = rvd_mspectr_dur       # 259
        numcep = 128
    else:
        dur = icp_mfcc_dur      # 50
        numcep = icp_numcep     # 40
    # disgust is somehow not in RAVDESS dataset, only neu, hap, sad, ang, fear
    n_classes = 5

    input_type = args.model_input
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    L_RATE = args.lr
    W_DECAY = args.w_decay

    train_dataset, test_dataset = load_data(dataset=args.dataset, input_type=input_type)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create an instance of our model
    if args.model_name == "gru":
        unsqueeze_needed = True
        model = Conv_GRU(input_shape=[BATCH_SIZE, 1, numcep, dur], kernel1=(3,3), kernel2=(2,2), hidden_size=32).to(device)
    elif args.model_name == "cnn":
        unsqueeze_needed = True
        if input_type == "mfcc" or input_type == "mspectr":
            model = CNN(n_classes=n_classes, kernel1=(3,3), kernel2=(2,2), numcep=numcep, dur=dur, \
                        input_type=input_type).to(device)
        else:                  # input_type = "spectr"
            unsqueeze_needed = False
            model = CNN_Spectr(n_classes=n_classes, input_size=spectr_dur, pool_size=2, num_filters=[8, 8, 8, 8], \
                           dropout=0.5, conv_size=3).to(device)
    elif args.model_name == "mlp-tbf":
        unsqueeze_needed = False
        input_type = "tbf"
        model = MLP(8, 256, 150, 4).to(device)
    elif args.model_name == "att_lstm":
        unsqueeze_needed = False
        input_type = "mfcc"
        model = AttentionLSTM(BATCH_SIZE, n_classes=n_classes, hidden_dim=100, emb_dim=dur)
    else:
        raise Exception("model_name parameter has to be one of \
        [gru|cnn|att_lstm] | [mspectr|mfcc|tbf|txt] | [iemocap|ravdess] | batch_size | number of epochs")

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    criterion = nn.CrossEntropyLoss()
    #criterion = t.nn.BCEWithLogitsLoss()

    # set up the optimizer
    optimizer = Adam(model.parameters(), lr=L_RATE, weight_decay=L_RATE) # weight_decay = 0.0001
    #optimizer = t.optim.SGD(model.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)

    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(model, args.model_name, args.dataset, criterion, optimizer, train_loader, test_loader, \
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
    plt.savefig('losses_{}.png'.format(args.model_name))

