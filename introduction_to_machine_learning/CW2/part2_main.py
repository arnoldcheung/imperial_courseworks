import pandas as pd
import part2_claim_classifier_alex as pm2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import tensorflow as tf




data_full=pd.read_csv('part2_data.csv')
X_raw=data_full.drop(columns=['made_claim', 'claim_amount']).values
y_raw=data_full.loc[:,'made_claim'].values.ravel()

model = pm2.ClaimClassifier()

X_train, X_test, Y_train, Y_test = train_test_split(X_raw, y_raw, test_size=0.1)

p = {'learning_rate': (0.0015),
     'first_node':(7),
     'second_node':(7),
     'batch_size': (32),
     'epochs': (20),
     'dropout': (0),
     'optimizer': (tf.keras.optimizers.Adam)}

# pm2.ClaimClassifierHyperParameterSearch(X_train,Y_train, model, p)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
#
# num_0 = 0
# num_1 = 0
# total_0 = 0
# total_1 = 0
# for i in range(Y_pred.shape[0]):
#     if round(Y_pred[i][0]) == Y_test[i]:
#         if Y_test[i] == 0:
#             total_0 += Y_pred[i]
#             num_0 += 1
#         elif Y_test[i] == 1:
#             total_1 += Y_test[i]
#             num_1 += 1
#
#     print(Y_pred[i], Y_test[i])
#
# print('Avg 0 = {}'.format(total_0/num_0))
# print('Avg 1 = {}'.format(total_1/num_1))


fpr, tpr, thresholds = roc_curve(Y_test, model.predict(X_test)[:,1])
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.2f)' % auc(fpr, tpr))
plt.legend(loc='best')
plt.show()