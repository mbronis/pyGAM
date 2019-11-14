

# https://codeburst.io/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f


import pandas as pd        
from pygam import LogisticGAM

from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#-----------------------------------------------------
#load the breast cancer data set

ds = load_breast_cancer()

X, y = ds.data, ds.target

#select first 6 features only
X = X[:,0:6]

selected_features = ds.feature_names[0:6]

#-----------------------------------------------------
#Fit a model with the default parameters
gam = LogisticGAM().fit(X, y)
gam.summary()

roc_auc_score(y,gam.predict_proba(X)) #0.994173140954495
gam.accuracy(X, y) #0.9560632688927944

#-----------------------------------------------------
# Explore and interpret individual features


plt.ion()
plt.rcParams['figure.figsize'] = (28, 8)

fig, axs = plt.subplots(1, X.shape[1])

for i, ax in enumerate(axs):
    XX=gam.generate_X_grid(term=i, meshgrid=True)
    pdep, confi = gam.partial_dependence(term=i, X=XX, meshgrid=True, width=.95)
    ax.plot(XX[0], pdep)
    ax.plot(XX[0], confi[:, 0], c='grey', ls='--')
    ax.plot(XX[0], confi[:, 1], c='grey', ls='--')
    ax.set_title(selected_features[i])
    
plt.show()


#-----------------------------------------------------
# Tuning Smoothness and Penalties


n_splines = [25, 6, 25, 25, 6, 4] 
lambda_ = 0.6
constraints = None

gam2 = LogisticGAM(
    constraints=constraints
    ,lam=lambda_
    ,n_splines=n_splines).fit(X, y)
gam2.summary()

roc_auc_score(y,gam2.predict_proba(X)) #0.991015274034142
gam2.accuracy(X, y) #0.9507908611599297


plt.ion()
plt.rcParams['figure.figsize'] = (28, 8)

fig, axs = plt.subplots(1, X.shape[1])

for i, ax in enumerate(axs):
    XX=gam2.generate_X_grid(term=i, meshgrid=True)
    pdep, confi = gam2.partial_dependence(term=i, X=XX, meshgrid=True, width=.95)
    ax.plot(XX[0], pdep)
    ax.plot(XX[0], confi[:, 0], c='grey', ls='--')
    ax.plot(XX[0], confi[:, 1], c='grey', ls='--')
    ax.set_title(selected_features[i])
    
plt.show()


#-----------------------------------------------------
# Grid search with pyGAM


#default in pyGAM grid search is lambda space of {'lam':np.logspace(-3,3,11)}

gam3 = LogisticGAM()
gam3.gridsearch(X, y)
gam3.summary