import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample, shuffle
from sklearn.metrics import auc, roc_curve, roc_auc_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class ClaimClassifier:
    def __init__(self,):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.ss = StandardScaler()
        self.base_classifier = None
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

        return  self.ss.fit_transform(X_raw)

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
        X_clean = self._preprocessor(X_raw)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_raw), y_raw)
        es = EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience = 5, mode='auto', restore_best_weights=True)
        # ARCHITECTURE OF OUR MODEL 
        
        def make_nn(hidden_layers=[7, 7], lrate=0.001):
            sgd = optimizers.SGD(lr=lrate)
            adam = optimizers.Adam(lr=lrate)
            he_init = he_normal()
            model = Sequential()
            for k in hidden_layers:
                model.add(Dense(k, activation='relu', kernel_initializer=he_init, bias_initializer='zeros'))
            model.add(Dense(1, activation='sigmoid', kernel_initializer=he_init, bias_initializer='zeros'))
            model.compile(loss='binary_crossentropy', optimizer=adam)
            return model

        self.base_classifier = KerasClassifier(make_nn, class_weight=class_weights, epochs=350, validation_split=0.1, batch_size=32, callbacks=[es])

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        
        self.base_classifier.fit(X_clean, y_raw)
        return self.base_classifier



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
        X_clean = self.ss.transform(X_raw)
        
        return  self.base_classifier.predict_proba(X_clean)
       

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


def ClaimClassifierHyperParameterSearch(X_train,Y_train, classifier, param):  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters.

    """

    model = classifier.base_classifier
    hp=GridSearchCV(model, param_grid= param, scoring = 'roc_auc')
    hp.fit(X_train,Y_train)
    #print('Best params',hp.best_params_)

    return  hp.best_params_

if __name__ == "__main__":
    data = pd.DataFrame(pd.read_csv('part2_data.csv'))
    data = shuffle(data)

    train_split_ind = int(round(data.shape[0] * 0.9))

    train_data = data.iloc[:train_split_ind, :]
    test_data = data.iloc[train_split_ind:, :]

    # data_majority = train_data[train_data.made_claim == 0]
    # data_minority = train_data[train_data.made_claim == 1]
    # data_minority_upsampled = resample(data_minority, replace=True, n_samples=data_majority.shape[0], random_state=123)
    # train_data = pd.concat([data_majority, data_minority_upsampled])

    train_data = train_data.values
    test_data = test_data.values

    train_X = train_data[:, :9].astype(np.float32)
    train_Y = train_data[:, -1].astype(np.float32)

    test_X = test_data[:, :9].astype(np.float32)
    test_Y = test_data[:, -1].astype(np.float32)

    classifier = ClaimClassifier()

    classifier.fit(train_X, train_Y)

    pred_Y = classifier.predict(test_X)

    classifier.evaluate_architecture(test_Y, pred_Y)



