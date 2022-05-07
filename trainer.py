import os.path

import numpy as np
import torch as t
import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report

#from tqdm.autonotebook import tqdm
from utils import EarlyStopping, get_datetime

#aibo_labels_dict = {'A': 0, 'E': 1, 'N': 2, 'P': 3, 'R': 4}
#ID_TO_CLASS = {v: k for k, v in aibo_labels_dict.items()}
#w2v_classes = list(ID_TO_CLASS.keys())

"""
class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 model_name,
                 dataset,
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation data set
                 test_dl=None,  # Test data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1,   # The patience for early stopping
                 unsqueeze_needed=True,
                 classnames=None):

        self._model = model
        self.model_name = model_name
        self.dataset = dataset
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._test_dl = test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self._unsqueeze_needed = unsqueeze_needed
        self.CLASS_TO_ID = classnames[0]
        self.ID_TO_CLASS = classnames[1]
        self.CLASSNAMES = classnames[2]

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    #def save_checkpoint(self, epoch):
     #   t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    #def restore_checkpoint(self, epoch_n):
    def restore_checkpoint(self):
        path = 'checkpoints/' + self.model_name + '_checkpoint_{}.ckp'.format(get_datetime())
        if os.path.exists(path):
            #ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
            ckp = t.load(path, 'cuda' if self._cuda else None)
            self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients / clear the gradients of all optimized variables
        self._optim.zero_grad()
        # -propagate through the network / forward pass: compute predicted outputs by passing inputs to the model
        output = self._model.forward(x)
        # -calculate the loss
        loss = self._crit(output, y)
        # -compute gradient by backprop / backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # -update weights / perform a single optimization step (parameter update)
        self._optim.step()
        # -return the loss
        return loss, output

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        pred = self._model.forward(x)
        # calculate the loss
        loss = self._crit(pred, y)
        # return the loss and the predictions
        return loss, pred

    def train_epoch(self):
        # set training mode / prepare model for training
        self._model.train()
        # iterate through the training set
        # clear lists to track next epoch
        total_loss = 0
        total_acc = 0
        for x, y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            if self._unsqueeze_needed:
                x = x.unsqueeze(1)
            # perform a training step
            loss, pred = self.train_step(x, y)
            total_loss += loss.item()
            #total_acc += accuracy_score(y.cpu().detach().numpy(), np.hstack(pred))
            #total_acc += accuracy_score(y.cpu(), pred.cpu() > 0.5)
        # calculate the average loss for the epoch and return it
        total_loss = total_loss / len(self._train_dl)
        #total_acc = total_acc / len(self._train_dl)
        #print("Train: loss: {}, accuracy: {}".format(total_loss, total_acc))
        print("Train: loss: {}".format(total_loss))
        return total_loss

    def val_test(self, test=False):
        # set eval mode / prepare model for evaluation
        self._model.eval()
        # disable gradient computation (disable autograd engine)
        t.no_grad()
        # iterate through the validation set
        # clear lists to track next epoch
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        if test:
            dataset = self._test_dl
        else:
            dataset = self._val_test_dl
        for x, y in dataset:
            # transfer the batch to the gpu if given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            if self._unsqueeze_needed:
                x = x.unsqueeze(1)

            # perform a validation step / forward pass: compute predicted outputs by passing inputs to the model
            loss, pred = self.val_test_step(x, y)       # pred.shape torch.Size([8, 5]) = bs, num_cl
            # calculate metrics for this iteration
            total_loss += loss.item()

            # deal with multilabel
            activation = t.nn.Softmax(dim=1)
            pred = activation(pred.data)
            pred = t.max(pred, 1)[1]    # choose maximum class index for the most predominant index
            # pred: tensor([4, 3, 2, 4, 0, 3, 4, 3])
            #pred = pred.cpu().detach()
            pred = pred.cpu().detach().numpy()

            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in list(self.ID_TO_CLASS.keys())}
            total_pred = {classname: 0 for classname in list(self.ID_TO_CLASS.keys())}
            # collect the correct predictions for each class
            for label, prediction in zip(y, pred):
                if label == prediction:
                    correct_pred[list(self.ID_TO_CLASS.keys())[label]] += 1
                total_pred[list(self.ID_TO_CLASS.keys())[label]] += 1

            # print accuracy for each class
            #for classname, correct_count in correct_pred.items():
             #   accuracy = 100 * float(correct_count) / total_pred[classname]
                #print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
              #  print("Accuracy for class {} is: {} %".format(classname, accuracy))

            #total_acc += accuracy_score(y.cpu(), pred.cpu() > 0.5)
            total_acc += accuracy_score(y.cpu().detach().numpy(), np.hstack(pred))
            #total_f1 += f1_score(y.cpu(), pred.cpu() > 0.5, average=None)
            total_f1 += f1_score(y.cpu().detach().numpy(), np.hstack(pred), average='weighted')
            # save the predictions and the labels for each batch

        # calculate the average loss and average metrics
        total_loss = total_loss / len(dataset)
        total_acc = total_acc / len(dataset)
        total_f1 = total_f1 / len(dataset)

        # return the loss and print the calculated metrics
        print("Test: loss: {}, accuracy: {}%, f-score: {}".format(total_loss, total_acc * 100, total_f1))
        #print(classification_report(y, pred, target_names=w2v_classes))
        t.enable_grad()
        return total_loss

    def fit(self, n_epochs):
        # to track the training loss as the model trains
        #train_losses = []
        # to track the validation loss as the model trains
        #valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        # store results
        #res = open('./results/' + self.model_name + '_results.txt', 'w')
        #res.write(50 * '=')
        #res.write('Model \n')
        #res.write(str(self._model) + '\n')

        # load the last checkpoint with the best model
        self.restore_checkpoint()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self._early_stopping_patience, verbose=True)

        for epoch in range(1, n_epochs + 1):
            # train the model
            train_loss = self.train_epoch()
            # validate the model
            valid_loss = self.val_test()

            # calculate average loss over an epoch
            #train_loss = np.average(train_losses)
            #train_loss = train_losses / len(self._train_dl)
            #valid_loss = np.average(valid_losses)

            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # early_stopping needs the validation loss to check if it has decreased,
            # if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self._model, self.model_name, self.dataset)

            if early_stopping.early_stop:
                print("Early stopping has been reached")
                break

            # load the last checkpoint with the best model
            #self._model.load_state_dict(t.load('checkpoint.pt'))
            #self.restore_checkpoint()

        # return model, avg_train_losses, avg_valid_losses
        #res.close()
        return avg_train_losses, avg_valid_losses

    def test(self):
        #avg_test_losses = []

        # load the last checkpoint with the best model
        self.restore_checkpoint()

        # initialize the early_stopping object
        #early_stopping = EarlyStopping(patience=self._early_stopping_patience, verbose=True)

        # validate the model
        test_loss = self.val_test()

        # calculate average loss over an epoch
        #train_loss = np.average(train_losses)
        #train_loss = train_losses / len(self._train_dl)
        #valid_loss = np.average(valid_losses)

        #avg_test_losses.append(test_loss)

            # early_stopping needs the validation loss to check if it has decreased,
            # if it has, it will make a checkpoint of the current model
            #early_stopping(test_loss, self._model, self.model_name, self.dataset)

            #if early_stopping.early_stop:
             #   print("Early stopping has been reached")
              #  break

        # return model, avg_train_losses, avg_valid_losses
        #res.close()
        #return avg_test_losses
        return test_loss
"""

#CLASS_TO_ID, ID_TO_CLASS, CLASSNAMES = get_classnames()

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 model_name,
                 dataset,
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation data set
                 test_dl=None,  # Test data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1,  # The patience for early stopping
                 unsqueeze_needed=True,
                 CLASS_TO_ID=None):
                 #ID_TO_CLASS=None,
                 #CLASSNAMES=None):

        self._model = model
        self.model_name = model_name
        self.dataset = dataset
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._test_dl = test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self._unsqueeze_needed = unsqueeze_needed

        self.CLASS_TO_ID = CLASS_TO_ID
        #self.ID_TO_CLASS = ID_TO_CLASS
        #self.CLASSNAMES = CLASSNAMES

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    # def save_checkpoint(self, epoch):
    #   t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    # def restore_checkpoint(self, epoch_n):
    def restore_checkpoint(self):
        path = 'checkpoints/' + self.model_name + '_checkpoint_{}.ckp'.format(get_datetime())
        if os.path.exists(path):
            # ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
            ckp = t.load(path, 'cuda' if self._cuda else None)
            self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients / clear the gradients of all optimized variables
        self._optim.zero_grad()
        # -propagate through the network / forward pass: compute predicted outputs by passing inputs to the model
        output = self._model.forward(x)
        # -calculate the loss
        # y=y.to(t.int64)

        loss = self._crit(output, y)
        # -compute gradient by backprop / backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # -update weights / perform a single optimization step (parameter update)
        self._optim.step()
        # -return the loss
        return loss, output

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        pred = self._model.forward(x)
        # calculate the loss
        # y=y.to(t.int64)

        loss = self._crit(pred, y)
        # return the loss and the predictions
        return loss, pred

    def train_epoch(self):
        # set training mode / prepare model for training
        self._model.train()
        # iterate through the training set
        # clear lists to track next epoch
        total_loss = 0
        total_acc = 0
        for x, y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            if self._unsqueeze_needed:
                x = x.unsqueeze(1)
            # perform a training step
            # print(y.dtype)
            loss, pred = self.train_step(x, y)
            total_loss += loss.item()
            # total_acc += accuracy_score(y.cpu().detach().numpy(), np.hstack(pred))
            # total_acc += accuracy_score(y.cpu(), pred.cpu() > 0.5)
        # calculate the average loss for the epoch and return it
        total_loss = total_loss / len(self._train_dl)
        # total_acc = total_acc / len(self._train_dl)
        # print("Train: loss: {}, accuracy: {}".format(total_loss, total_acc))
        print("Train: loss: {}".format(total_loss))
        return total_loss

    def val_test(self, mode=False):
        # set eval mode / prepare model for evaluation
        self._model.eval()

        all_preds = []
        # disable gradient computation (disable autograd engine)
        with t.no_grad():
            # iterate through the validation set
            # clear lists to track next epoch
            total_loss = 0
            total_acc = 0
            total_f1 = 0
            if mode:
                dataset = self._test_dl
            else:
                dataset = self._val_test_dl
            for x, y in dataset:
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                if self._unsqueeze_needed:
                    x = x.unsqueeze(1)

                # perform a validation step / forward pass: compute predicted outputs by passing inputs to the model
                loss, pred = self.val_test_step(x, y)  # pred.shape torch.Size([8, 5]) = bs, num_cl
                # calculate metrics for this iteration
                total_loss += loss.item()

                # deal with multilabel
                activation = t.nn.Softmax(dim=1)
                pred = activation(pred.data)
                pred = t.max(pred, 1)[1]  # choose maximum class index for the most predominant index
                # pred: tensor([4, 3, 2, 4, 0, 3, 4, 3])
                # pred = pred.cpu().detach()
                pred = pred.cpu().detach().numpy()

                # prepare to count predictions for each class
                correct_pred = {classname: 0 for classname in list(self.CLASS_TO_ID.values())}
                total_pred = {classname: 0 for classname in list(self.CLASS_TO_ID.values())}
                # collect the correct predictions for each class
                for label, prediction in zip(y, pred):
                    if label == prediction:
                        correct_pred[list(self.CLASS_TO_ID.values())[label]] += 1
                    total_pred[list(self.CLASS_TO_ID.values())[label]] += 1

                # print accuracy for each class
                # for classname, correct_count in correct_pred.items():
                #   accuracy = 100 * float(correct_count) / total_pred[classname]
                # print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
                #  print("Accuracy for class {} is: {} %".format(classname, accuracy))

                # total_acc += accuracy_score(y.cpu(), pred.cpu() > 0.5)
                total_acc += accuracy_score(y.cpu().detach().numpy(), np.hstack(pred))
                # total_f1 += f1_score(y.cpu(), pred.cpu() > 0.5, average=None)
                total_f1 += f1_score(y.cpu().detach().numpy(), np.hstack(pred), average='weighted')

                # save the predictions and the labels for each batch
                all_preds = np.hstack([all_preds, pred])

            # calculate the average loss and average metrics
            total_loss = total_loss / len(dataset)
            total_acc = total_acc / len(dataset)
            total_f1 = total_f1 / len(dataset)

            # return the loss and print the calculated metrics
            if mode:
                print("Test: loss: {}, accuracy: {}%, f-score: {}".format(total_loss, total_acc * 100, total_f1))
            else:
                print("Validation: loss: {}, accuracy: {}%, f-score: {}".format(total_loss, total_acc * 100, total_f1))

        t.enable_grad()

        if mode:
            return total_loss, all_preds
        else:
            return total_loss

    def fit(self, n_epochs):
        # to track the training loss as the model trains
        # train_losses = []
        # to track the validation loss as the model trains
        # valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        # store results
        # res = open('./results/' + self.model_name + '_results.txt', 'w')
        # res.write(50 * '=')
        # res.write('Model \n')
        # res.write(str(self._model) + '\n')

        # load the last checkpoint with the best model
        self.restore_checkpoint()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self._early_stopping_patience, verbose=True)

        for epoch in range(1, n_epochs + 1):
            # train the model
            train_loss = self.train_epoch()
            # validate the model
            valid_loss = self.val_test(mode=False)

            # calculate average loss over an epoch
            # train_loss = np.average(train_losses)
            # train_loss = train_losses / len(self._train_dl)
            # valid_loss = np.average(valid_losses)

            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            """
            # print training/validation statistics
            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)
            """

            # early_stopping needs the validation loss to check if it has decreased,
            # if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self._model, self.model_name, self.dataset)

            """
            # use the save_checkpoint function to save the model for each epoch
            save_flag = self._early_stopping_cb.step(l_dev)

            if save_flag:
                res.write(50 * '=')
                res.write('Epoch: ' + str(self.epoch) + ' Training Loss :' + str(l_train) + ' Development Loss :' + str(
                    l_dev))
                Trainer.save_checkpoint(self, self.epoch + 1, model_name)
                self.epoch_n = self.epoch + 1
            """

            if early_stopping.early_stop:
                print("Early stopping has been reached")
                break

            # load the last checkpoint with the best model
            # self._model.load_state_dict(t.load('checkpoint.pt'))
            # self.restore_checkpoint()

        # return model, avg_train_losses, avg_valid_losses
        # res.close()
        return avg_train_losses, avg_valid_losses

    def test(self):

        # load the last checkpoint with the best model
        self.restore_checkpoint()

        # test the model
        test_loss, all_preds = self.val_test(mode=True)
        return test_loss, all_preds

    def get_dataset(self):
        return self.dataset


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.RdPu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # if normalize:
    #    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #   print("Normalized confusion matrix")
    # else:
    #   print('Confusion matrix, without normalization')
    #print('Confusion matrix')
    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('res/conf_matrix_{}.png'.format(Trainer.get_dataset()))
    plt.show()
    return cm


def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels

"""
def display_results(y_test, pred):
    # pred = np.argmax(pred_probs, axis=-1)
    # one_hot_true = one_hot_encoder(y_test, len(pred), len(CLASS_TO_ID))
    one_hot_true = one_hot_encoder(y_test, len(pred), len(CLASS_TO_ID))
    print('Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred)))
    print('Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro')))
    print('Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro')))
    print('Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro')))
    # if cm:
    plot_confusion_matrix(confusion_matrix(y_test, pred), classes=list(ID_TO_CLASS.values()))
    print(classification_report(y_true=y_test, y_pred=pred, target_names=CLASSNAMES))
"""


def display_results(y_test, pred, dataset, model, model_input, CLASS_TO_ID):
    # pred = np.argmax(pred_probs, axis=-1)
    # one_hot_true = one_hot_encoder(y_test, len(pred), len(CLASS_TO_ID))
    #one_hot_true = one_hot_encoder(y_test, len(pred), len(CLASS_TO_ID))
    acc = 'Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred))
    fscore = 'Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro'))
    prec = 'Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro'))
    rec = 'Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro'))
    metrics = [acc, fscore, prec, rec]
    # if cm:
    cr = classification_report(y_true=y_test, y_pred=pred, target_names=list(CLASS_TO_ID.keys()))
    cm = plot_confusion_matrix(confusion_matrix(y_test, pred), classes=list(CLASS_TO_ID.keys()))
    f = open('res/report_{{}_{}_{}.png'.format(dataset, model, model_input), 'w')
    f.write(
        'RESULTS\n\nMetrics\n\n{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(metrics, cr, cm))
    f.close()
    print(cr)
    return cm


"""
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses = []
        test_losses = []
        epoch = 0

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self._early_stopping_patience, verbose=True)

        last_test_loss = float('inf')
        stop_iterations = 0
        while True:
            # train for an epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            test_loss = self.val_test()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if epoch % 10 == 0:
                # store checkpoint every 10 epochs
                self.save_checkpoint(epoch)
                # TODO: change saving checkpoints

            epoch += 1
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self._early_stopping_patience > 0:
                if self._early_stopping_patience <= stop_iterations:
                    break
                # update stop iterations
                stop_iterations += 1
                if (last_test_loss - test_loss) > 1e-4:
                    stop_iterations = 0
                last_test_loss = test_loss
            elif epoch == epochs:
                break

        # store checkpoint when ending the training
        self.save_checkpoint(epoch)

        # return the losses for both training and validation
        return train_losses, test_losses
    """


"""
# Model training and saving best model
best_accuracy = 0.0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
           images = Variable(images.cuda())
           labels = Variable(labels.cuda())

        images = images.unsqueeze(1)
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        images = images.unsqueeze(1)

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy
"""
