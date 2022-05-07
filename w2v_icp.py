import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from models import MLP
from trainer import Trainer
from utils import get_device


ft_df = np.load("/Users/el/embrace/data/data_wav2vec2/icp/" + "icp_ft.npy")
#hs_df = np.load("/Users/el/embrace/data/data_wav2vec2/icp/" + "icp_hs.npy")
labels_df = np.load("/Users/el/embrace/data/data_wav2vec2/icp/" + "labels.npy")

labels_dict = {'ang': 1, 'exc': 3, 'fea': 2, 'fru': 4, 'hap': 3, 'neu': 0, 'oth': 6, 'sad': 4, 'sur': 5, 'xxx': 6}
ID_TO_CLASS = {v: k for k, v in labels_dict.items()}
w2v_classes = list(ID_TO_CLASS.keys())

lb_df = np.array([labels_dict[letter] for letter in labels_df])

my_x = ft_df
my_y = lb_df
split_index = 5630
all_indexes = list(range(my_x.shape[0]))
test_indexes = all_indexes[(split_index + 1):]
x_test = my_x[test_indexes]
y_test = my_y[test_indexes]

train_indexes = all_indexes[:(split_index + 1)]
x_train = my_x[train_indexes]
y_train = my_y[train_indexes]

eval_split_index = 934
all_indexes = list(range(x_train.shape[0]))
eval_indexes = all_indexes[(eval_split_index + 1):]
eval_train_indexes = all_indexes[:(eval_split_index + 1)]

x_train_eval = x_train[eval_indexes]
y_train_eval = y_train[eval_indexes]
x_eval = x_train[eval_train_indexes]
y_eval = y_train[eval_train_indexes]

# x_train_eval=np.vstack(x_train_eval).astype(np.float)
# y_train_eval=np.vstack(y_train_eval).astype(np.str)

# print(y_train_eval.shape, y_eval.shape, y_test.shape)       # (4696,) (935,) (1407,)
# print(x_train_eval.shape, x_eval.shape, x_test.shape)       # (4696, 3072) (935, 3072) (1407, 3072)

train_dataset = TensorDataset(torch.tensor(x_train_eval), torch.tensor(y_train_eval))  # create your dataset
#train_dataset = TensorDataset(torch.from_numpy(x_train_eval).float(), torch.from_numpy(y_train_eval).float())
eval_dataset = TensorDataset(torch.tensor(x_eval), torch.tensor(y_eval))  # create your dataset
test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(dataset=eval_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

#BATCH_SIZE = 100
device = get_device()
#unsqueeze_needed = False
#input_type = "tbf"


"""
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            #nn.Linear(3072, 1000),     # icp_ft.npy
            nn.Linear(6144, 1000),
            nn.ReLU(),
            nn.Linear(1000, 7)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.layers(x)
        return x
"""


# defining model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, out_dim=2):
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
        out = self.linear_1_bn(out)
        out = F.relu(out)

        out = self.linear_2(out)
        out = self.linear_2_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.175, training=self.training)

        out = self.linear_out(out)
        return out

#model = MLP(6144, 3000, 1000, 7)
model = MLP(3072, 3000, 1000, 7)
print(model)

criterion = nn.CrossEntropyLoss()

# set up the optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0)

trainer = Trainer(model, "w2v", "icp", criterion, optimizer, train_loader, test_loader, \
                  cuda=torch.cuda.is_available(), early_stopping_patience=20)

# go, go, go... call fit on trainer
res = trainer.fit(100)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses_{}.png'.format("w2v"))

"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 15

for epoch in range(epochs):
    model.train()

    train_losses = []
    valid_losses = []
    for i, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (i * 128) % (128 * 100) == 0:
            print(f'{i * 128} / 50000')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            valid_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))

    accuracy = 100 * correct / total
    valid_acc_list.append(accuracy)
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%' \
          .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), accuracy))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
ax1.plot(mean_train_losses, label='train')
ax1.plot(mean_valid_losses, label='valid')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='best')

ax2.plot(valid_acc_list, label='valid acc')
ax2.legend()

model.eval()
test_preds = torch.LongTensor()

for i, images in enumerate(test_loader):
    outputs = model(images)

    pred = outputs.max(1, keepdim=True)[1]
    test_preds = torch.cat((test_preds, pred), dim=0)



# model = MLP(8, 256, 150, 4).to(device)


# some hyper parameters
# BATCH_SIZE = 8
# EPOCHS = 10
# LEARN_RATE = 0.001     # 0.0001 performs worse
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0


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
"""
