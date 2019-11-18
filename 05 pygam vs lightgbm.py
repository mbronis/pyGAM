
import pandas as pd
import numpy as np

import pygam as pg
import lightgbm as lgb

import matplotlib.pyplot as plt

#load dataset
redwine_url = 'https://raw.githubusercontent.com/ianshan0915/medium-articles/master/data/redwine-quality.csv'
redwine = pd.read_csv(redwine_url)

redwine.describe()
redwine['quality'].value_counts(sort=False)


#gam

X=redwine.drop('quality', axis=1).values
y=redwine['quality']
feature_names=redwine.columns[:-1]

lams=np.logspace(-10,20, 20)
gam=pg.PoissonGAM().gridsearch(X,y,lam=lams)
gam.summary()


#lightgbm

params = {
    'objective': 'poisson'
    ,'metric': 'poisson'
#    ,'num_trees': 200
#    ,'early_stopping_round': 25
    ,'max_depth': 5
    ,'num_leaves': 10
    ,'min_data_in_leaf': 200
    ,'max_delta_step': 0.3
    ,'learning_rate': 0.1    
}

train = lgb.Dataset(
    redwine.drop('quality', axis=1)
    ,redwine['quality'])

lgbm_cv=lgb.cv(params
    ,train
    ,num_boost_round=500
    ,early_stopping_rounds=25
    ,verbose_eval=25
    ,show_stdv=False
    ,nfold=5
)
best_round=len(lgbm_cv['poisson-mean'])

lgbm=lgb.LGBMRegressor(objective='poisson', metric='poisson'
    , num_boost_round=best_round)
lgbm.fit(redwine.drop('quality', axis=1), redwine['quality'])


ax = lgb.plot_importance(lgbm, max_num_features=100)
plt.show()

#summ and compare

lgbm_pred=pd.Series(lgbm.predict(X))
gam_pred=pd.Series(gam.predict(X))

y.hist()
gam_pred.hist()
lgbm_pred.hist()

y.value_counts(sort=False)
round(gam_pred,0).astype(int).value_counts(sort=False)
round(lgbm_pred,0).astype(int).value_counts(sort=False)

sum((y-y)/y+np.log(y))
sum((y-gam_pred)/gam_pred+np.log(gam_pred))
sum((y-lgbm_pred)/lgbm_pred+np.log(lgbm_pred))


#pdp plots

from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(lgbm, X, list(range(0,len(redwine.columns)-1)))



plt.figure()
fig, axs = plt.subplots(1,11,figsize=(40, 8))
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX,   width=.95)[1], c='r', ls='--')
    if i == 0:
        ax.set_ylim(-30,30)
    ax.set_title(feature_names[i])


