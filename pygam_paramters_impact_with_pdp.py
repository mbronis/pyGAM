

'''
TODO:
0) proper import and naming ot terms
00) usage of 0 variance terms
1) fit gam on continous and categorical data
2) integers as numerical and categorical data
3) gridsearch on lambdas
4) grid search on type of terms

Insights:
-all data required in numerical format (ndarray),
    factors need to be transformed to integers
'''

#import pygam lib

import numpy as np
import pandas as pd

from pygam.datasets import wage


#utils

def crude_strCat_to_int(ar,findex):
    '''
    check if feature is a string
    if so replaces its unique values with an integer 
    coresponding to lexicographical order
    '''

    if isinstance(ar[0,findex], str):
        ar[:,findex]=np.unique(ar[:,findex], return_inverse=True)[1]

# load dataset (as pd.DataFrame) => describe features

df=wage(return_X_y=False)
df.describe(include='all')

'''
type of terms:
1) int/category ['year']
2) int          ['age']
3) continous    ['logwage']
4) category     ['sex','maritl','race','education','religion','jobclass','health','health_ins']
'''


#prep X and y

features=['year', 'age', 'education']

X=df[features].values
crude_strCat_to_int(X,2)
y=df['wage'].values


# test different types of term on categorical feature
# term types: spline (default), linear effect, factor, spline with categorical dtype

from pygam import LinearGAM, s,f,l

gam1=LinearGAM(s(0)+s(1)+s(2)).fit(X,y)
gam2=LinearGAM(s(0)+s(1)+l(2)).fit(X,y)
gam3=LinearGAM(s(0)+s(1)+f(2)).fit(X,y)
gam4=LinearGAM(s(0)+s(1)+s(2, dtype='categorical')).fit(X,y)

gams=[gam1,gam2,gam3,gam4]
terms_names=['spline','linear','factor','categorical spline']

###########################################
#compare pdp

import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (28, 28)
fig, axs = plt.subplots(4, X.shape[1], sharey='row')

for r, axr in enumerate(axs):
    gam=gams[r]
    for i, ax in enumerate(axr):
        XX=gam.generate_X_grid(term=i, meshgrid=True)
        pdep, confi = gam.partial_dependence(term=i, X=XX, meshgrid=True, width=.95)
        ax.plot(XX[0], pdep)
        ax.plot(XX[0], confi[:, 0], c='grey', ls='--')
        ax.plot(XX[0], confi[:, 1], c='grey', ls='--')
        if r==0:
            ax.set_title(features[i])
        if i==0:
            ax.set_ylabel(terms_names[r], size='large')
