

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
4) categorical  ['sex','maritl','race','education','religion','jobclass','health','health_ins']
'''


#prep X and y

X=df[['year', 'age', 'education']].values
crude_strCat_to_int(X,2)

y=df['wage'].values


# run linear Gam with defaults

from pygam import LinearGAM, s,f,l

gam=LinearGAM().fit(X,y)
gam.summary()


gam2=LinearGAM(s(0)+s(1)+f(2)).fit(X,y)
gam2.summary()


gam3=LinearGAM(s(0)+s(1)+s(2, dtype='categorical')).fit(X,y)
gam3.summary()


gam4=LinearGAM(s(0)+s(1)+l(2)).fit(X,y)
gam4.summary()


#compare pdp

import matplotlib.pyplot as plt

XX=gam.generate_X_grid(term=2, meshgrid=True)

pdep = gam.partial_dependence(term=2, X=XX, meshgrid=True)
pdep2 = gam2.partial_dependence(term=2, X=XX, meshgrid=True)
pdep3 = gam3.partial_dependence(term=2, X=XX, meshgrid=True)
pdep4 = gam4.partial_dependence(term=2, X=XX, meshgrid=True)


plot(XX[0], pdep)
set_title(selected_features[i])






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