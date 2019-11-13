

# https://codeburst.io/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f


import pandas as pd        
from pygam import LogisticGAM
from sklearn.datasets import load_breast_cancer

#load the breast cancer data set
data = load_breast_cancer()

#keep first 6 features only
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']

X = pd.DataFrame(data.data, columns=data.feature_names)[selected_features]
y = pd.Series(data.target)

#Fit a model with the default parameters
gam = LogisticGAM().fit(X, y)
gam.summary()

from sklearn.metrics import roc_auc_score
roc_auc_score(y,gam.predict_proba(X)) #0.994173140954495
gam.accuracy(X, y) #0.9560632688927944

# plot each term

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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