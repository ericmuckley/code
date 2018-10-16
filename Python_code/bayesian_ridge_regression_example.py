import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

from sklearn.linear_model import BayesianRidge, LinearRegression


#%% set up data 

df2 = df.values

np.random.seed(0)

n_samples, n_features = len(df2), len(df2[0])-1


#training data
X = df2[:,:-1] 

# target data
y = df2[:,-1]



#%% fit regression models

svr_rbf = SVR(kernel='rbf', C=1e10, gamma=100000)



print('training model 1...')
y_rbf = svr_rbf.fit(X, y).predict(X)




#%% look at results
lw = 2
plt.semilogx(X[:,1], y, color='darkorange', label='data')
plt.semilogx(X[:,1], y_rbf, color='navy', label='RBF model')
#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()