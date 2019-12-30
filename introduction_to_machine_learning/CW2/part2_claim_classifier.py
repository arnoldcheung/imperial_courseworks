
import numpy as np
import pandas as pd
#import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import StratifiedKFold
#from imblearn.under_sampling import RandomUnderSampler
#mport matplotlib.pyplot as plt
#from torch.autograd import Variable
import tensorflow as tf
#import talos

import pickle


class Model:

    def __init__(self,parameters):
        self.model = tf.keras.models.Sequential()
        self.NormalInitializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.25)
        self.epochs=int(parameters['epochs'])
        self.first_node=int(parameters['first_node'])
        self.second_node=int(parameters['second_node'])
        self.dropout=float(parameters['dropout'])
        self.optimizers=parameters['optimizer']
        self.lr=float(parameters['learning_rate'])
        self.batch_size=int(parameters['batch_size'])



    def contruct_model(self):
        self.model.add(tf.keras.layers.Dense(self.first_node, activation='relu', kernel_initializer=self.NormalInitializer,
                                             bias_initializer='zeros'))
        self.model.add(tf.keras.layers.Dropout(self.dropout))
        self.model.add(tf.keras.layers.Dense(self.second_node, activation='relu', kernel_initializer=self.NormalInitializer,
                                             bias_initializer='zeros'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=self.NormalInitializer,
                                             bias_initializer='zeros'))
        return self.model

    def compile_model(self):
        return self.model.compile(optimizer=self.optimizers(lr=self.lr), loss='binary_crossentropy')

    def fit_model(self, X_train, Y_train,class_weights,X_val,Y_val):
        return self.model.fit(X_train,Y_train,class_weight=class_weights,epochs=self.epochs,validation_data=(X_val,Y_val),batch_size=self.batch_size)

    def model_summary(self):
        return self.model.summary()

    def model_predict(self, X_test):
        return self.model.predict(X_test)

class ClaimClassifier:
    def __init__(self, parameters):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        # Hyper-parameters

        # self.calibrate = calibrate_probabilities
        # self.removed_indices = None

        # Neural network
        self.model = Model(parameters)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.model.lr)
        #self.callbacks= tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.005, patience = 5, mode='auto', restore_best_weights=True)

        pass

    def _preprocessor(self, X_raw, drop_na=False):
        """Data preprocessing function.
9
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

        full_features = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus', 'vh_sale_begin', 'vh_sale_end',
                         'vh_value', 'vh_speed']

        data = pd.DataFrame(X_raw, columns=full_features)
        #
        # if drop_na:
        # # Removing rows having any NA value and remembering the lines when droped
        #     self.removed_indices = data.isna().any(axis=1).values
        #     data.dropna(inplace=True)
        # else:
        #     # replace NA by the mean of the column
        #     data.fillna(data.mean(), inplace=True)
        X=data.values

        #Normalizing data
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)


        return X_scaled

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
        # pass
        #ros = RandomUnderSampler()

        X = self._preprocessor(X_raw)
        #X_over, y_raw_over = ros.fit_resample(X,y_raw)
        X_train, X_val, Y_train, Y_val = train_test_split(X, y_raw, test_size=0.1)




        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_raw), y_raw)
        self.model.contruct_model()

        # Compile model
        self.model.compile_model()

        # Enforces early stop if val_loss behaves in a certain way

        # Fit model. Progress is not lost when you rerun this cell, e.g. if you run it manually twice it's like you did 2*num_epochs of training.
        self.model.fit_model(X_train, Y_train,class_weights, X_val, Y_val)
        #self.model.model_summary()

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


        # YOUR CODE HERE

        X_clean = self._preprocessor(X_raw)


        return self.model.model_predict(X_clean)

    def evaluate_architecture(self, Y_test, Y_predict):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """

        predictions = [round(value) for value in Y_predict.flatten()]
        accuracy=accuracy_score(Y_test,predictions)
        con_mat = confusion_matrix(Y_test, predictions)
        print('Confusion matrix : \n',con_mat)
        print('Accuracy ',accuracy)


        fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.2f)' % auc(fpr, tpr))
        # plt.legend(loc='best')
        # plt.show()
        AUC=auc(fpr,tpr)
        return accuracy,AUC

    def save_model(self):
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)


def ClaimClassifierHyperParameterSearch(parameters,model):  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """

    return  # Return the chosen hyper parameters

# #
# data = pd.DataFrame(pd.read_csv('part2_data.csv'))
#
# data = data.values
#
# # print(data)
#
# X = data[:, :9].astype(np.float32)
# Y = data[:, -1].astype(np.float32)
#
#classifier = ClaimClassifier()

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=23)

# X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.25)
# X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25)

# classifier.fit(X_train, Y_train)
#
# Y_pred = classifier.predict(X_test)
#
# classifier.evaluate_architecture(Y_test, Y_pred)


# #Cross_validation
# cv = StratifiedKFold(n_splits=10, shuffle=True)
# aucscores = []
# accscores =[]
# i=1
# for train, test in cv.split(X, Y):
#     classifier = ClaimClassifier()
#     classifier.fit(X[train], Y[train])
#     Y_pred = classifier.predict(X[test])
#     acc, aucc=classifier.evaluate_architecture(Y[test], Y_pred)
#     aucscores.append(aucc)
#     accscores.append(acc)
#     print("Accuracy of {0} for the fold {1}".format(auc,i))
#     i+=1
#
# print("Mean AUC {0}".format(sum(aucscores)/len(aucscores)))
# print("Mean Accurracy {0}".format(sum(accscores)/len(accscores)))

#
# def model_nn(X_train,Y_train, X_val,Y_val, params):
#     classifier = ClaimClassifier(params)
#     history=classifier.fit(X_train, Y_train)
#     return history,classifier
#
#
# p = {'learning_rate': (0.5, 5, 10),
#      'first_node':[4,7, 8, 16, 32, 64],
#      'second_node':[4,7, 8, 16, 32, 64],
#      'batch_size': (32, 50, 64),
#      'epochs': [150],
#      'dropout': (0, 0.5, 5),
#      'optimizer': [tf.keras.optimizers.Adam, tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop]}
#
# classifier = ClaimClassifier(p)
