#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:44:54 2019

@author: aleksander
"""
import pandas as pd
import part3_pricing_alex as pm
import part2_claim_classifier_alex as pm2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

#data_full=pd.read_csv('part3_data.csv')
#X_raw=data_full.drop(columns=['made_claim', 'claim_amount']).values
#y_raw=data_full.loc[:,'made_claim'].values.ravel()
#claims_raw=data_full.loc[:,'claim_amount'].values.ravel()
#model = pm.PricingModel(calibrate_probabilities=True)
#
#X_train, X_test, Y_train, Y_test = train_test_split(X_raw, y_raw, test_size=0.1, random_state=0)
#
#
#model.fit(X_train, Y_train, claims_raw)
#model.predict_claim_probability(X_test)
#
#
#fpr, tpr, thresholds = roc_curve(Y_test, model.predict_claim_probability(X_test)[:,1])
#plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.2f)' % auc(fpr, tpr))
#plt.legend(loc='best')
#plt.show()

data_full=pd.read_csv('part2_data.csv')
X_raw=data_full.drop(columns=['made_claim', 'claim_amount']).values
y_raw=data_full.loc[:,'made_claim'].values.ravel()

model = pm2.ClaimClassifier()

X_train, X_test, Y_train, Y_test = train_test_split(X_raw, y_raw, test_size=0.1, random_state=0)


model.fit(X_train, Y_train)
model.predict(X_test)


fpr, tpr, thresholds = roc_curve(Y_test, model.predict(X_test)[:,1])
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.2f)' % auc(fpr, tpr))
plt.legend(loc='best')
plt.show()