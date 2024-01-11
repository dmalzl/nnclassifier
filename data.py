import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_encoding_dictionary(encoder):
    return {i: cat for i, cat in enumerate(encoder.categories_[0])}


def one_hot_encoding(labels):
    # encode categorical data with one hot encoding
    encoder = OneHotEncoder(
        handle_unknown = 'ignore',
        sparse_output = False
    )
    labels = labels.reshape(-1, 1)
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    return encoded_labels, get_encoding_dictionary(encoder)


def one_hot_decoding(encoded_predictions, encoding):
    encoded_labels = torch.argmax(encoded_predictions, 1).numpy()
    return [encoding[label] for label in encoded_labels]


def pandas_to_tensor(data):
    return torch.from_numpy(data).type(torch.float)


def prepare_data_for_torch(features, labels):
    '''
    expects features and labels as pandas.DataFrame/pandas.Series object
    and return pytorch tensor of features and one-hot encoded labels
    '''
    encoded_labels, encoding = one_hot_encoding(labels.values)
    features_tensor = torch.from_numpy(features.values).type(torch.float)
    encoded_labels_tensor = torch.from_numpy(encoded_labels).type(torch.float)

    return features_tensor, encoded_labels_tensor, encoding
