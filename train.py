import tqdm
import torch

import pandas as pd

from torch import nn
from sklearn.model_selection import train_test_split
        

def score(y_pred, y_test):
    # simple prediction accuracy
    return (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()


def train_step(model, X_train, y_train, optimizer, loss_function):
    # performs the train step
    model.train()                               # sets model into train mode
    y_pred = model(X_train)                     # predict labels from train data
    loss = loss_function(y_pred, y_train)       # computes loss i.e. how far our predictions are away from the true output
    optimizer.zero_grad()                       # zeros gradient to numerical problems
    loss.backward()                             # performs backpropagation
    optimizer.step()                            # take a step towards lower loss
    accuracy = score(y_pred, y_train)           # compute accuracy
    return float(loss), float(accuracy)


def eval_step(model, X_test, y_test, loss_function):
    # evaluates model based on test data
    model.eval()
    y_pred = model(X_test)
    loss = loss_function(y_pred, y_test)
    accuracy = score(y_pred, y_test)
    return float(loss), float(accuracy)


def train_test_histories_to_dataframe(train_history, test_history, value_name):
    # converts recorded losses into seaborn compatible dataframes
    history_frame = pd.DataFrame(
        {
            'train': train_history,
            'test': test_history
        }
    )
    history_frame['epoch'] = range(len(train_history))
    history_frame = history_frame.melt(
        value_vars = ['train', 'test'],
        var_name = 'eval_time',
        value_name = value_name,
        id_vars = ['epoch']
    )
    return history_frame


def train_model(model, X, y, n_epochs = 10):
    """
    trains a given model on the provided data for n_epochs.
    train test split is performed automatically here 10% of the data are held out
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size = 0.1,
        stratify = y
    )

    # initialize Adam optimizer, this is what performs the gradient decent computation
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    # initialize loss function, cross entropy best for multiclass prediction
    loss_function = nn.CrossEntropyLoss()

    # record losses and accuracies during training
    train_loss_history, train_accuracy_history = [], []
    test_loss_history, test_accuracy_history = [], []

    # this is just a progressbar and helps us keep everything tidy
    with tqdm.trange(n_epochs, unit = 'epoch', mininterval = 0) as epoch_bar:
        # perform train and eval steps for each epoch
        # each time we do this the neural net parameters 
        # get updated according to the gradient computed from the loss
        # this essentially means we evaluate how far we are away from an "optimal"
        # model and take a predefined step towards this optimum. Then we reset and repeat
        for _ in epoch_bar:      
            train_loss, train_accuracy = train_step(
                model, 
                X_train, 
                y_train, 
                optimizer, 
                loss_function
            )
                
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            
            test_loss, test_accuracy = eval_step(
                model,
                X_test,
                y_test,
                loss_function
            )
            test_loss_history.append(test_loss)
            test_accuracy_history.append(test_accuracy)
            
            epoch_bar.set_postfix(
                loss = train_loss,
                accuracy = train_accuracy
            )

    loss_histories = train_test_histories_to_dataframe(
        train_loss_history,
        test_loss_history,
        'cross_entropy'
    )
    
    accuracy_histories = train_test_histories_to_dataframe(
        train_accuracy_history,
        test_accuracy_history,
        'accuracy'
    )

    return loss_histories, accuracy_histories
