# MACHINE LEARNING CLASSIFIERS USING SKLEARN
import csv, sys, numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#%%

var1 = np.array([1,2,3,4,10,11,12,13])
var2 = np.array([-1,-2,3,-4,20,-2,.18,-2.5])
classes = np.array([0,0,0,0,1,1,1,1])
#create 2 classes:

meshstep = .1  # step size in the mesh

#%% insert data to be used as columns for input matrix 
inputlist =(
    var1,
    var2)
#insert data to be used as columns for target matrix
targetlist = (
    classes.astype(int))
  #  presscol)
##################

#see if input/target lists are multi-dimensional and convert to matrices

if len(np.shape(inputlist)) > 1:
    inputmatrix = np.array(inputlist).T #create input matrix
else:
    inputmatrix = inputlist.reshape(len(inputlist),) #change to '),1)' if necessary
    
if len(np.shape(targetlist)) > 1:
    targetmatrix = np.array(targetlist).T #create target matrix
else:
    targetmatrix = targetlist.reshape(len(targetlist),) #change to '),1)' if necessary

#split into random train and test sets
inp_train, inp_test, tar_train, tar_test = train_test_split(
    inputmatrix, targetmatrix, train_size=0.75, random_state = 0)

X = inputmatrix
y = targetmatrix

#%%

#Start classifier
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Lin. Disc. Analysis",
         "Quad. Disc. Analysis"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# preprocess dataset, split into training and test part
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 #design meshgrid space
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, meshstep),
                     np.arange(y_min, y_max, meshstep))

#%% plot the raw dataset
cm = plt.cm.RdBu #red-blue colormap
cm_bright = ListedColormap(['#FF0000', '#0000FF']) #multiple brightnesses

ax = plt.subplot(2, 5, 1) #start subplot
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_title('Data')


i = 1 #set first iteration number
for name, clf in zip(names, classifiers): # iterate over each classifier and plot
    ax = plt.subplot(2, 5, i+1) #start subplot
    clf.fit(X_train, y_train) #fit classifier
    score = clf.score(X_test, y_test) #find score for each classifier

    # Plot the decision boundary. assign a color to each mesh point
    # [x_min, m_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)
    
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    '''
    # save all meshes
    meshname = 'binaryclassmesh' + format(i) + '.csv'
    with open(meshname, "wb") as output:
        writer = csv.writer(output)
        writer.writerows(list(reversed(Z)))
    '''
    i += 1

#figure.subplots_adjust(left=.02, right=.98)
plt.show()
