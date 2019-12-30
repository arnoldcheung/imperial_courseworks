import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
import pickle

import nn_lib as nn

import torch


class ClaimClassifier:
    def __init__(self, input_dim, neurons, activations):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.network = nn.MultiLayerNetwork(input_dim, neurons, activations)
        self.trainer = nn.Trainer(
            network=self.network,
            batch_size=30,
            nb_epoch=1000,
            learning_rate=0.01,
            loss_fun="cross_entropy",
            shuffle_flag=True,
        )


    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        X: numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE

        # data = X_raw.drop(columns=drop_cols)
        #
        # X = data.loc[:, ~data.columns.isin(label_cols)].values
        # Y = data.loc[:, data.columns.isin(label_cols)].values.ravel()
        #
        # split_idx = int(0.8 * len(X))
        #
        # x_train = X[:split_idx]
        # y_train = Y[:split_idx]
        # x_val = X[split_idx:]
        # y_val = Y[split_idx:]

        prep = nn.Preprocessor(X_raw)
        x_train_pre = prep.apply(X_raw)

        return x_train_pre  # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded
        y_raw : numpy.ndarray (optional)
            A one dimensional numpy array, this is the binary target variable

        Returns
        -------
        ?
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE

        X_clean = self._preprocessor(X_raw)
        self.trainer.train(X_clean, y_raw)

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        X_clean = self._preprocessor(X_raw)
        predictions = self.network(X_clean)

        return predictions  # YOUR NUMPY ARRAY

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self):
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)


def ClaimClassifierHyperParameterSearch():  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """

    return  # Return the chosen hyper parameters


if __name__ == "__main__":

    drop_cols = ['claim_amount']
    label_cols = ['made_claim']

    data = pd.read_csv('part2_data.csv')

    data = data.drop(columns=drop_cols)

    X = data.loc[:, ~data.columns.isin(label_cols)].values
    Y = data.loc[:, data.columns.isin(label_cols)].values

    split_idx = int(0.8 * len(X))

    x_train = X[:split_idx]
    y_train = Y[:split_idx]
    x_val = X[split_idx:]
    y_val = Y[split_idx:]

    num_features = x_train.shape[1]

    classifier = ClaimClassifier(num_features, [30, 30, 1], ['relu', 'relu', 'sigmoid'])
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_val)

    print(predictions)

