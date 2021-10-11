import torch as t
from sklearn.metrics import f1_score, accuracy_score
# from tqdm.autonotebook import tqdm
import numpy as np
from utils import EarlyStopping, get_datetime


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 model_name,
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1, # The patience for early stopping
                 unsqueeze_needed=True):
        self._model = model
        self.model_name = model_name
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self._unsqueeze_needed = unsqueeze_needed

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    #def save_checkpoint(self, epoch):
     #   t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    #def restore_checkpoint(self, epoch_n):
    def restore_checkpoint(self):
        #ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        ckp = t.load('checkpoints/' + self.model_name + 'checkpoint_{}.ckp'.format(get_datetime()), \
                     'cuda' if self._cuda else None)
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

    def val_test(self):
        # set eval mode / prepare model for evaluation
        self._model.eval()
        # disable gradient computation (disable autograd engine)
        t.no_grad()
        # iterate through the validation set
        # clear lists to track next epoch
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        for x, y in self._val_test_dl:
            # transfer the batch to the gpu if given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            if self._unsqueeze_needed:
                x = x.unsqueeze(1)

            # perform a validation step / forward pass: compute predicted outputs by passing inputs to the model
            loss, pred = self.val_test_step(x, y)
            # calculate metrics for this iteration
            total_loss += loss.item()

            # deal with multilabel
            activation = t.nn.Softmax(dim=1)
            pred = activation(pred.data)
            pred = t.max(pred, 1)[1]    # choose maximum class index for the most predominant index
            pred = pred.cpu().detach().numpy()

            #total_acc += accuracy_score(y.cpu(), pred.cpu() > 0.5)
            total_acc += accuracy_score(y.cpu().detach().numpy(), np.hstack(pred))
            #total_f1 += f1_score(y.cpu(), pred.cpu() > 0.5, average=None)
            total_f1 += f1_score(y.cpu().detach().numpy(), np.hstack(pred), average='weighted')
            # save the predictions and the labels for each batch

        # calculate the average loss and average metrics
        total_loss = total_loss / len(self._val_test_dl)
        total_acc = total_acc / len(self._val_test_dl)
        total_f1 = total_f1 / len(self._val_test_dl)

        # return the loss and print the calculated metrics
        print("Test: loss: {}, accuracy: {}%, f-score: {}".format(total_loss, total_acc, total_f1))
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
            early_stopping(valid_loss, self._model, self.model_name)

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
                print("Early stopping")
                break

            # load the last checkpoint with the best model
            #self._model.load_state_dict(t.load('checkpoint.pt'))
            self.restore_checkpoint()

        # return model, avg_train_losses, avg_valid_losses
        #res.close()
        return avg_train_losses, avg_valid_losses

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
