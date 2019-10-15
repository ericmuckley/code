# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:18:54 2018
@author: a6q
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import linear_model

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer




def config_plot(xlabel='x', ylabel='y', size=16,
               setlimits=False, limits=[0,1,0,1]):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    #set axis limits
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))


def get_hop_dic(hop_file):
    #create dictionary which holds all hop information using table "hop_file"
    
    #initialize empty dictionary
    hop_dic = {}
    #loop through descriptors file to populate hop dictionary
    for i in range(len(hop_file)):
        #initialize entry for each hop 
        hop0 = hop_file['hop'].iloc[i]
        hop_dic[hop0] = {}
    
        #add entries for each hop decriptor
        for descriptor in hop_file.columns[1:]:
            hop_dic[hop0][descriptor] = hop_file[descriptor].iloc[i]
            
            #separate some descriptors into lists
            if type(hop_dic[hop0][descriptor])==str and ',' in hop_dic[hop0][descriptor]:
                hop_dic[hop0][descriptor] = hop_dic[hop0][descriptor].split(',')
    return hop_dic




def normalize_columns(array0):
    #normalize each column of a 2D array from 0-1
    array = np.copy(array0)
    for col in range(len(array[0])):
        array[:,col] = array[:,col] - np.amin(array[:,col])
        array[:,col] = array[:,col] / np.amax(array[:,col])
    return array








def plot_hyperplane(clf, min_x, max_x, label=None):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else: raise ValueError
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    model = OneVsRestClassifier(SVC(kernel='linear'))
    model.fit(X, Y)
    print('model score: '+str(model.score(X,Y)))

    plt.subplot(2, 2, subplot)
    plt.title(title)

    #plot scatter of training
    #plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    
    
    #plot points for each sample
    for jj in range(len(aromas)): 
        plt.scatter(X[np.where(Y[:, jj]), 0], X[np.where(Y[:, jj]), 1], s=1+jj, label=aromas[jj])
        plot_hyperplane(model.estimators_[jj], min_x, max_x)#, label='c'+str(jj))

    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 4:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc='upper right', fontsize=12)









#%% import data to create dictionary of hop characteristics
hop_dic_file = pd.read_table('exp_data\\hops_descriptors.txt')
hop_dic = get_hop_dic(hop_dic_file)


#%% import data for hop sensor response for use in target array
hop_mq_file0 = pd.read_csv('exp_data\\hop_alldata.csv')
hop_mq_file = hop_mq_file0.loc[hop_mq_file0['sample'] %2 == 0]



#%% separate hop mq file by day for mapping aromas to natural language
#hop_mq_file_by_day = hop_mq_file.loc[(hop_mq_file['day']) == 3]
#hop_mq_file_drift = hop_mq_file.iloc[::150]

day0 = 1
#X0 = hop_mq_file[['mq2', 'mq4', 'mq5', 'mq7', 'mq2/mq7', 'mq4/mq7',
#                  'mq5/mq7', 'mq2/mq5', 'mq4/mq5', 'hop']]

X0 = hop_mq_file[['mq2', 'mq4',
                  'mq5', 'mq7', 
                  #'mq2/mq7', 'mq4/mq7',
                  #'mq5/mq7', 'mq2/mq5', 'mq4/mq5',
                  'hop'
                  ]].loc[(hop_mq_file['day']) == day0]



hops = np.array(np.unique(np.array(X0['hop']))).astype(str)
avg_hop_dic = {}
#get average X values for each hop
for hop in hops: avg_hop_dic[hop] = X0.loc[X0['hop'] == hop].mean()


#get X vector for aroma training
#triple X so that each  aroma combination is accounted for
X = np.zeros((len(X0)*3, len(X0.columns)-1))
for i in range(0, len(X)-1, 3):
    X[i, :] = X0.iloc[int(i/3), :-1]
    X[i+1, :] = X0.iloc[int(i/3), :-1]
    X[i+2, :] = X0.iloc[int(i/3), :-1]

#get y vector for aroma training
y0 = [hop_dic[X0['hop'].iloc[i]]['aroma'][0] for i in range(len(X0))]
y1 = [hop_dic[X0['hop'].iloc[i]]['aroma'][1] for i in range(len(X0))]
y2 = [hop_dic[X0['hop'].iloc[i]]['aroma'][2] for i in range(len(X0))]
y = np.reshape(np.column_stack((y0, y1, y2)), (-1, 1))

#run regression
logreg = linear_model.LogisticRegression(C=1e5) #specify regression parameters
logreg.fit(X,y) 
aroma_classes = logreg.classes_


#get aroma prediction results
aroma_prediction_dic = {}
for hop in avg_hop_dic:
    test_vec0 = np.reshape(np.array(avg_hop_dic[hop]), (1,-1))
    prediction0 = logreg.predict_proba(test_vec0)
    prediction0 = np.reshape(prediction0,(len(logreg.classes_,)))
    prediction0 = np.round(prediction0, 4)*100
    prediction0 = np.stack((prediction0, logreg.classes_)).T
    aroma_prediction_dic[hop] = prediction0


all_predictions = np.zeros((14,9))
for i in range(len(aroma_prediction_dic)):
    
    all_predictions[:,i] = (aroma_prediction_dic[hops[i]][:,0]).astype(float)






#%% plot MQ7 vs MQ2 boundaries

'''
hop_integers = {'amarillo':1, 'cascade':2, 'centennial':3,
                'citra':4, 'columbus':5, 'liberty':6,
                'magnum':7, 'simcoe':8, 'willamette':9}


#for day0 in [1,2,3,4,18]:
for day0 in [1]:  
    
    print(day0)
    X = np.array(hop_mq_file[['mq2', 'mq7']].loc[(hop_mq_file['day']) == day0])
    y = np.array(hop_mq_file['hop'].loc[(hop_mq_file['day']) == day0])
    
    
    #plt.scatter(hop_mq_file['mq2'].loc[(hop_mq_file['day']) == day0],
    #            hop_mq_file['mq7'].loc[(hop_mq_file['day']) == day0],
    #            s=2)
    
    
    colors = "bgrcmykwb"

    # shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    # standardize
    #mean = X.mean(axis=0)
    #std = X.std(axis=0)
    #X = (X - mean) / std
    
    h = .005  # step size in the mesh
    
    clf = SVC(C=1e12, kernel='linear', verbose=True).fit(X,y)#SGDClassifier(alpha=0.001, n_iter=100).fit(X, y)
    
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - .01, X[:, 0].max() + .01
    y_min, y_max = X[:, 1].min() -.01, X[:, 1].max() + .01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    plt.set_cmap(plt.cm.Paired)
    
    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    for jj in range(len(Z)):
        Z[jj] = hop_integers[Z[jj]]
    #
    Z = Z.astype(float)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.set_cmap(plt.cm.Paired)
    cs = plt.contourf(xx, yy, Z)
    plt.axis('tight')
    
    # Plot also the training points
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color)#, label=iris.target_names[i])
    plt.title('Day '+str(day0)+', score = '+format(clf.score(X,y)))
    plt.axis('tight')
    
    
    
    plt.show()
    
    df_copy = pd.DataFrame(data=np.column_stack((X,y)), columns=['mq2', 'mq7', 'hop'])
    
'''




#%% predict flavors example
'''
import numpy as np
import random

classes = ['sweet', 'sour', 'bitter', 'piney', 'citrusy', 'skunky', 'earthy']

# generate fake input data
inputmatrix = np.random.random((30,5)) #input matrix with 5 inputs, 30 samples

#make list of target classess
targets = [random.choice(classes) for i in range(len(inputmatrix))]

# perform regression
from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1e5) #specify regression parameters

X = inputmatrix 
y = targets


logreg.fit(X,y) #run regression
logreg.classes_ #summon the class names

testvector = [inputmatrix[3]] #select new data for testing

prediction = logreg.predict_proba(testvector) #assign score to each class
prediction = np.reshape(prediction,(len(logreg.classes_,))) #reshape to match class vector shape
prediction = np.round(prediction,2)*100 #convert to %

result = np.stack((prediction, logreg.classes_)).T #attach flavors to results of prediction

for i in range(len(prediction)):
    print(format(prediction[i]) + '% = ' + format(logreg.classes_[i]))
'''
    


#%%



'''
#get table, list, and one-hot encoding of all aromas
aroma_table = [hop_dic[hop_mq_file.iloc[i]['hop']]['aroma'] for i in range(len(hop_mq_file))]
one_hot = MultiLabelBinarizer()
#one-hot encoding of all aromas
Y = one_hot.fit_transform(aroma_table)
aromas = np.array(one_hot.classes_).astype(str)
'''

#%% configure feature array
'''
feature_names = ['mq2', 'mq4', 'mq5', 'mq7',
                'mq2/mq7', 'mq4/mq7', 'mq5/mq7',
                'mq2/mq4', 'mq2/mq5', 'mq4/mq5']
#X = normalize_columns(hop_mq_file[feature_names])
X = normalize_columns(hop_mq_file[feature_names])
'''
#%% create a dataframe with all data 
'''
df_all = pd.concat([pd.DataFrame(data=X, columns=feature_names),
                    pd.DataFrame(data=Y, columns=aromas)], axis=1)
df_all[['day','hop']] = hop_mq_file[['day','hop']]

'''
#%% find centroids of each aroma over time

'''
days = np.unique(df_all['day'])

#create dictionary to hold all centroid data
centroid_dic = {aroma: np.zeros((len(days), 8)) for aroma in aromas}

#find mean and std of each centroid over time and record in dictionary
for j in range(len(aromas)):
    for i in range(len(days)):
        mean0 = df_all.loc[(df_all[aromas[j]]==1) & (df_all['day']==days[i])].mean()
        std0 = df_all.loc[(df_all[aromas[j]]==1) & (df_all['day']==days[i])].std()
        
        centroid_dic[aromas[j]][i][0:4] = mean0[['mq2', 'mq4', 'mq5', 'mq7']]
        centroid_dic[aromas[j]][i][4:8] = std0[['mq2', 'mq4', 'mq5', 'mq7']]

#turn dictionary keys into dataframes so columns are labeled
for aroma in aromas: 
    centroid_dic[aroma] = pd.DataFrame(data=centroid_dic[aroma],
                        columns=['mq2', 'mq4', 'mq5', 'mq7',
                                 'std2', 'std4', 'std5', 'std7'])
    #rearrange column order so mean and std columns are adjacent
    centroid_dic[aroma] = centroid_dic[aroma][['mq2', 'std2',
                                                'mq4', 'std4',
                                                'mq5', 'std5',
                                                'mq7', 'std7']]

'''    
   
  
    
    
    
#%%
'''
plt.figure(figsize=(8, 6))

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")
plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()

'''




#%% generate training array
'''
df = np.empty((0, len(hop_file.columns)-1))
for i in range(len(hop_file)):
    #append every descriptor
    for j in range(3):
        df = np.vstack((df, np.append(hop_file.iloc[i,:-2],
                                      hop_dic[hop_file.iloc[i]['hop']]['aroma'][j])))
              
hop_vec = df[:,0].astype(str)
aroma_vec = df[:,-1].astype(str)
features_full = df[:,1:-1].astype(float)        
'''  


#%%




#df = pd.DataFrame(data=df, columns=hop_dic_file.columns[:-1])
#aroma_vec = np.array(df['aroma']).astype(str)


#mq_array = df[sensor_names].values

#mq_array = normalize_columns(mq_array)


#%%

#X = mq_array.astype(float)
#y, y_labels = pd.factorize(aroma_vec)




#%% loop over features
'''
features = features_full[::3,:]    
feature_pairs = ([0,1], [0,2], [0,3], [1,2], [1,3], [2,3])



#np.random.seed(13) # fix the seed on each iteration
n_estimators=20
models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=n_estimators),
          #SVC(kernel='rbf'),
          #SVC(kernel='poly', degree=2),
          #SVC(kernel='poly', degree=3)]
          ]


i = 0
for model in models:
    print(i)
    X = mq_array.astype(float)
    y, y_labels = pd.factorize(aroma_vec)

    model.fit(X, y)

    model_title = str(type(model))

    print(model_title+' score: '+str( model.score(X, y)))

    if 'linear' in model_title: coeffs = model.coef_
    else: coeffs = model.feature_importances_
        
        
    print('feature importance: '+str(coeffs))

    plt.plot(coeffs, label=model_title)
    plt.legend()
    config_plot('Feature column', 'Feature importance')
    
    i+=1

'''




#%% loop pver pairs of features to plot them
'''



# Parameters
n_classes = len(np.unique(aroma_vec))

cmap = ListedColormap(['r', 'y', 'g', 'c', 'b', 'm', 'w', 'k',
                                         '0.3','0.6', '0.9'])
plot_step = 0.5  # fine step width for decision surface contours
plot_step_coarser = 1  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration





plot_idx=1
for pair in feature_pairs:
    for model in models:
        X = mq_array[:,pair]*5
        
        y, y_labels = pd.factorize(aroma_vec)

        
        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        plt.subplot(len(feature_pairs), len(models), plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=6,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'g', 'c', 'b', 'm', 'w', 'k',
                                         '0.3','0.6', '0.9']),
                    edgecolor='k', s=40)
        plot_idx += 1  # move on to the next plot in sequence
plt.gcf().set_size_inches(8,12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.legend()
plt.show()    

'''

    
    
    
    
  
#%% get correlation matrix
''' 

features = features_full[::3,:]

corr_mat = np.empty((len(features[0]), len(features[0])))

for col1 in range(len(features[0])):
    for col2 in range(len(features[0])):
        corr0 = np.corrcoef(features[:,col1], features[:,col2])[0,1]
        corr_mat[col1][col2] =  corr0

corr_flat = np.unique(np.ndarray.flatten(corr_mat))
'''

#%% normalize features
'''
features_full_norm = np.zeros_like(features_full)
for i in range(len(features_full[0])):
    arr = features_full[:,i]
    arr = arr - np.amin(arr)
    arr = arr/np.amax(arr)
    features_full_norm[:,i] = arr

#offset identical adjacent samples 

for i in range(1, len(features_full_norm)):
        features_full_norm[i] += 0.001*i
'''




