from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from models import Conv_GRU, CNN, CNN_Spectr
from prepare_data import load_data, MFCC_DUR as mfcc_dur, DATA_DIR
from preprocess_audio import NUMCEP as numcep

DATA_DIR = "/Users/el/embrace/data/"
# DATA_DIR = "/home/yq36elyb/data/"
CHECKPOINT_DIR = "/Users/el/embrace/checkpoints/"
# CHECKPOINT_DIR = "/home/yq36elyb/checkpoints/"

# some hyper parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARN_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0      # 0.8


def train_step(config, checkpoint_dir=CHECKPOINT_DIR, data_dir=DATA_DIR):
    #net = CNN(num_classes=4)
    net = Conv_GRU(input_shape=[BATCH_SIZE, 1, numcep, mfcc_dur], hidden_size=32)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"],\
                           weight_decay=config["w_decay"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_dataset, test_dataset = load_data("mfcc")

    test_abs = int(len(train_dataset) * 0.8)
    train_subset, val_subset = random_split(
        train_dataset, [test_abs, len(train_dataset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = inputs.unsqueeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                inputs = inputs.unsqueeze(1)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device).unsqueeze(1), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples=10, max_num_epochs=100, gpus_per_trial=2):
    data_dir = DATA_DIR
    load_data(data_dir)
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "w_decay": tune.choice([0.0, 0.3, 0.7, 0.9]),
        "batch_size": tune.choice([16, 32, 64, 128, 256, 512])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["lr", "w_decay", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_step, data_dir=DATA_DIR),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Conv_GRU(best_trial.config["lr"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=100, gpus_per_trial=2)


"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from cnn import *

param_grid = [{'batch_size': [1, 10, 100, 1000]},{'learn_rate': [0.1, 0.01, 0.001, 0.0001]}]

#clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='f1_macro')
model = CNN()
clf = RandomizedSearchCV(model, param_grid, cv=5, scoring='f1_macro')

clf.fit(X_train, y_train)

print(clf.best_params_)

print(clf.cv_results_)

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    print(param, score)
"""
