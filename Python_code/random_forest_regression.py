# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:17:02 2018

@author: a6q
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR


#%% organize data

samples = 30

snum = np.arange(samples)/10

X = np.array([snum, snum, snum, snum]).T


y = snum

train_size = 20
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]



for i in range(len(X[0])):
    plt.plot(X[:,i], label=i)
plt.title('features')
plt.legend()
plt.show()


plt.plot(snum[:train_size], y_train, label='train')
plt.plot(snum[train_size:], y_test, label='test')
plt.legend()
plt.title('targets')
plt.show()




#%%

print('BOOSTED GRADIENT REGRESSION')

boosts = 400

gbr = GBR(n_estimators=boosts,
                                                          max_depth=4,
                                                          min_samples_split=2,
                                                          learning_rate=0.01)
        
gbr.fit(X_train, y_train)
mse = mean_squared_error(y_test, gbr.predict(X_test))
print("MSE: %.4f" % mse)  

print(gbr.predict(X_test))
  
# Plot training deviance

# compute test set deviance
test_score = np.zeros((boosts), dtype=np.float64)

for i, y_pred in enumerate(gbr.staged_predict(X_test)):
    test_score[i] = gbr.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(boosts)+1,gbr.train_score_, 'b-', label='train deviance')
plt.plot(np.arange(boosts)+1,test_score, 'r-', label='test deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations', fontsize=15)
plt.ylabel('Deviance', fontsize=15)


# Plot feature importance
gbr_feature_importance = gbr.feature_importances_
# make importances relative to max importance
gbr_feature_importance = 100.0 * (gbr_feature_importance / gbr_feature_importance.max())

sorted_idx = np.argsort(gbr_feature_importance)



pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, gbr_feature_importance[sorted_idx], align='center')
plt.yticks(pos, sorted_idx)
plt.xlabel('Relative Importance', fontsize=15)

plt.show()







print('------------------------------------------------')
    
#%%  random forest regression

print('RANDOM FOREST REGRESSION')

rfr = RFR(max_depth=2, random_state=0)
rfr.fit(X_train, y_train)

rfr_importances = rfr.feature_importances_

rfr_std = np.std([tree.feature_importances_ for tree in rfr.estimators_],
             axis=0)

rfr_importance_indices = np.argsort(rfr_importances)[::-1]

plt.title("Feature importances")
plt.bar(range(X.shape[1]), rfr_importances[rfr_importance_indices],
       color="r", yerr=rfr_std[rfr_importance_indices], align="center")
plt.xticks(range(X.shape[1]), rfr_importance_indices)
plt.xlim([-1, X.shape[1]])
plt.ylim([0, 1])
plt.show()

print(rfr_importances)

print(rfr.predict(X_test))


print('------------------------------------------------')