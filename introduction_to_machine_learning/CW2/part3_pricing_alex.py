from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
#    classifier = classifier.fit(X_train, y_train)
    classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = None # ADD YOUR BASE CLASSIFIER HERE
        self.ss = StandardScaler()

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded   ??
            

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        full_features = ['id_policy', 'pol_bonus', 'pol_coverage', 'pol_duration', 'pol_sit_duration', 'pol_pay_freq', 'pol_payd', 'pol_usage',
       'pol_insee_code', 'drv_drv2', 'drv_age1', 'drv_age2', 'drv_sex1', 'drv_sex2', 'drv_age_lic1', 'drv_age_lic2', 'vh_age', 'vh_cyl',
       'vh_din', 'vh_fuel', 'vh_make', 'vh_model', 'vh_sale_begin', 'vh_sale_end', 'vh_speed', 'vh_type', 'vh_value', 'vh_weight',
       'town_mean_altitude', 'town_surface_area', 'population', 'commune_code', 'canton_code', 'city_district_code', 'regional_department_code']

        full_data = pd.DataFrame(X_raw, columns=full_features)

        # Droping some columns we don't want to use
        data = full_data.drop(columns=['id_policy', 'commune_code', 'canton_code', 'city_district_code','regional_department_code', 'pol_insee_code', 'vh_make', 'vh_model', 'drv_drv2'])
        
        # replacing categories in pol_pay_freq by numbers of payments in a year
        values_pol_pay_freq = {'Yearly':1, 'Monthly':12, 'Biannual':0.5, 'Quarterly':4}
        data.pol_pay_freq = data.pol_pay_freq.replace(values_pol_pay_freq)

        # One Hot Encoding
        cat_cols = ['pol_coverage', 'pol_payd', 'pol_usage', 'drv_sex1', 'vh_fuel', 'vh_type']
        data = pd.get_dummies(data, prefix=cat_cols, columns=cat_cols, dummy_na=False)
        data = pd.get_dummies(data, prefix=['drv_sex2'], columns=['drv_sex2'], dummy_na=True)
        

        # replace NA by the median of the column
        data.fillna(data.median(), inplace=True)

        # returns an array of features
        return data.values


    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self.ss.fit_transform(self._preprocessor(X_raw))
        
        #Compute class weights and define callback for early stopping
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_raw), y_raw)
        es = EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience = 5, mode='auto', restore_best_weights=True)
        # ARCHITECTURE OF OUR MODEL 
        
        def make_nn(hidden_layers=[5,5], lrate=0.0005, random_seed=0):
            sgd = optimizers.SGD(lr=lrate)
            he_init = he_normal(seed=random_seed)
            model = Sequential()
            for k in hidden_layers:
                model.add(Dense(k, activation='relu', kernel_initializer=he_init, bias_initializer='zeros'))
            model.add(Dense(1, activation='sigmoid', kernel_initializer=he_init, bias_initializer='zeros'))
            model.compile(loss='binary_crossentropy', optimizer=sgd)
            return model

        self.base_classifier = KerasClassifier(make_nn, class_weight=class_weights, epochs=350, validation_split=0.1, batch_size=64, callbacks=[es])

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier.fit(X_clean, y_raw)
        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        X_clean = self.ss.transform(self._preprocessor(X_raw))
        return  self.base_classifier.predict_proba(X_clean) # return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

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
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)
