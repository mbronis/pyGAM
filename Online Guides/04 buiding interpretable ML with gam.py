#https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#load dataset
redwine_url = 'https://raw.githubusercontent.com/ianshan0915/medium-articles/master/data/redwine-quality.csv'
redwine = pd.read_csv(redwine_url)

redwine.describe()

redwine['quality'].value_counts(sort=False)
redwine.hist('quality')

X=redwine.drop('quality', axis=1).values
y=redwine['quality']
feature_names=redwine.columns[:-1]


#build linear and poisson gam

from pygam import PoissonGAM, LinearGAM

lams=np.logspace(-10,10,10)

poiss=PoissonGAM().gridsearch(X, y, lam=lams)
poiss.summary()

lin=LinearGAM().gridsearch(X,y,lam=lams)
lin.summary()


plt.figure()
fig, axs = plt.subplots(1,11,figsize=(40, 8))
for i, ax in enumerate(axs):
    XX = poiss.generate_X_grid(term=i)
    ax.plot(XX[:, i], poiss.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], poiss.partial_dependence(term=i, X=XX,   width=.95)[1], c='r', ls='--')
    if i == 0:
        ax.set_ylim(-30,30)
    ax.set_title(feature_names[i])


plt.figure()
fig, axs = plt.subplots(1,11,figsize=(40, 8))
for i, ax in enumerate(axs):
    XX = lin.generate_X_grid(term=i)
    ax.plot(XX[:, i], lin.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], lin.partial_dependence(term=i, X=XX,   width=.95)[1], c='r', ls='--')
    if i == 0:
        ax.set_ylim(-30,30)
    ax.set_title(feature_names[i])

pd.DataFrame(poiss.predict(X), columns=['poiss']).hist('poiss')
pd.DataFrame(lin.predict(X), columns=['lin']).hist('lin')
