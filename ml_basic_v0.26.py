
'''
v25 (v04 modified in notebook with tag
added read_csv_with_mappings
features importance table to cv output
print feature importances table
'''


import time
import datetime

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, ParameterGrid
from sklearn.metrics import SCORERS, mean_squared_error, roc_auc_score, log_loss

# import lightgbm as lgbm
from lightgbm import LGBMClassifier


pd.options.display.max_rows=2000

lgbm_parameters_synonyms={
    'n_estimators':['n_estimators','num_boost_round','num_iterations','num_trees','num_rounds', 'n_rounds', 'nrounds', 'rounds']
    ,'num_leaves':['num_leaves']
    ,'max_depth':['max_depth']
    ,'learning_rate':['learning_rate','learn_rate','eta']
    ,'min_child_samples':['min_child_samples','min_data_in_leaf','min_data', 'min_leaf']
    ,'min_child_weight':['min_child_weight','min_sum_hessian_in_leaf','min_hessian']
    ,'min_split_gain':['min_split_gain','min_gain_to_split']
    ,'colsample_bytree':['colsample_bytree','feature_fraction']
    ,'reg_alpha':['reg_alpha','lambda_l1']
    ,'reg_lambda':['reg_lambda','lambda_l2']
    ,'random_state': ['random_state','seed']
    ,'n_jobs':['n_jobs','num_threads', 'nthreads', 'cores']
    ,'device_type':['device_type','device']
    ,'gpu_device_id':['gpu_device_id']
    ,'gpu_platform_id':['gpu_platform_id']
    ,'max_tree_output':['max_tree_output']
    ,'min_data_per_group':['min_data_per_group']
    ,'max_cat_to_onehot':['max_cat_to_onehot']
    ,'max_cat_threshold':['max_cat_threshold']
    ,'cat_smooth':['cat_smooth']
    ,'cat_l2':['cat_l2']
    ,'max_bin':['max_bin']
    ,'min_data_in_bin':['min_data_in_bin']
    ,'feature_contri':['feature_contri','feature_contrib','fc','fp','feature_penalty']
}





def LGBMClassifier_binary(booster_params={}):
    # wrapper on LGBMClassifier method allowing to pass arguments as a dictionary: parameter_name:param_value
    # it allow for use of synonyms defined in external dictionary
    
    #setting up default valuest
    local_params={
        'n_estimators':100
        ,'num_leaves':31
        ,'max_depth':-1
        ,'learning_rate':0.1
        ,'min_child_samples':20
        ,'min_child_weight':1e-3
        ,'min_split_gain':0
        ,'colsample_bytree':1
        ,'reg_alpha':0.0
        ,'reg_lambda':0.0
        ,'random_state':1
        ,'n_jobs':8
        ,'device_type':'cpu'
        ,'gpu_device_id':0
        ,'gpu_platform_id':0
        ,'max_tree_output':0.0
        ,'min_data_per_group':100
        ,'max_cat_to_onehot':4
        ,'max_cat_threshold':32
        ,'cat_smooth':10.0
        ,'cat_l2':10.0
        ,'max_bin':255
        ,'min_data_in_bin':3   
        ,'feature_contri':None
    }
    
    for local_param in local_params:
        for param_syn in lgbm_parameters_synonyms[local_param]:
            if param_syn in booster_params:
                local_params[local_param]=booster_params[param_syn]
                break
                
    lgbm = LGBMClassifier(
        # sklearn parameters
        objective = 'binary'
        ,boosting ='gbdt'
        ,n_estimators=local_params['n_estimators']
        ,num_leaves=local_params['num_leaves']
        ,max_depth=local_params['max_depth']
        ,learning_rate=local_params['learning_rate']
        ,min_child_samples=local_params['min_child_samples']
        ,min_child_weight=local_params['min_child_weight']
        ,min_split_gain=local_params['min_split_gain']
        ,colsample_bytree=local_params['colsample_bytree']
        ,reg_alpha=local_params['reg_alpha']
        ,reg_lambda=local_params['reg_lambda']
        ,random_state=local_params['random_state']
        ,n_jobs=local_params['n_jobs']
        ,device_type=local_params['device_type']
        ,gpu_device_id=local_params['gpu_device_id']
        ,gpu_platform_id=local_params['gpu_platform_id']
        ,feature_contri=local_params['feature_contri']
        
        # lgbm parameters
        ,max_tree_output=local_params['max_tree_output']
        ,min_data_per_group=local_params['min_data_per_group']
        ,max_cat_to_onehot=local_params['max_cat_to_onehot']
        ,max_cat_threshold=local_params['max_cat_threshold']
        ,cat_smooth=local_params['cat_smooth']
        ,cat_l2=local_params['cat_l2']
        ,max_bin=local_params['max_bin']
        ,min_data_in_bin=local_params['min_data_in_bin']
    )
    return lgbm

def LGBMClassifier_binary_cv(
    booster_params={}, 
    X=None, 
    y=None, 
    eval_metric=['auc'], 
    early_stopping_rounds=25, 
    verbose=False,
    feature_name='auto', 
    categorical_feature='auto', 
    feature_remove=None,
    fold_column=None, 
    nfold=5, #number of folds to create if no 'folds' provided
    stratified=True, #Whether to create stratified folds
    return_estimators=True, #Whether to return the estimators fitted on each split.
    return_single_estimator=False, #Whether to train and return estimator trained on whole X with mean best_rounds.
    random_state=1
):
    # TODOS:
    #   -fold vector
    #   -metrics only from sklearn.metrics.SCORERS => so to calculate all_oof_preds error with same metric as cv eval
    #   -
    #   -
    #   -
    total_time=time.time()
    # 1 assert arguments
    assert not X is None, 'Error: X is None.'
    assert not y is None, 'Error: y is None.'
    assert X.shape[0]==len(y), 'Error: Different rows num in X and y.'
    assert feature_name == 'auto' or len(feature_name)>0, "Error (Arg): 'feature_name' is empty"
    assert feature_name == 'auto' or set(feature_name).issubset(set(X.columns)), \
        "Error (Arg): 'feature_name' must be subset of X columns: " + str(set(feature_name)-set(X.columns))
    
    
    # 2 filter features    
    if feature_name == 'auto':
        feature_name=list(X.columns)
    else:
        feature_name=list(set(X.columns) & set(feature_name))
    if not feature_remove is None:
        feature_name=list(set(feature_name)-set(feature_remove))
    if not fold_column is None:
        feature_name=list(set(feature_name)-set([fold_column]))        
        X.reset_index(drop=True, inplace=True)
        X_Fold=X[[fold_column]]

    X=X[feature_name]
    
    if categorical_feature != 'auto': categorical_feature=list(set(categorical_feature) & set(feature_name))
        
#     if not fold_column is None: X_Fold=X[[fold_column]]
#     if feature_name != 'auto': X=X[feature_name]
#     if not feature_remove is None:
#         feature_X=X.columns
#         feature_left=list(set(feature_X)-set(feature_remove))
#        
#     if categorical_feature != 'auto': categorical_feature=list(set(categorical_feature) & set(feature_left))
    
    
#     if feature_name=='auto': feature_name=X.columns
#     if not feature_remove is None: feature_name=list(set(feature_name)-set(feature_remove))
#     assert len(feature_name)>0, "Error (Arg): No features left in 'X'."
#     X=X[feature_name]
#     if categorical_feature != 'auto': categorical_feature=list(set(categorical_feature) & set(feature_name))
    
    # 3 create placeholders for results
    fold_best_iterations = []
    fold_best_scores = []
    fold_evals_results = []
    fold_train_times = []
    fold_feature_importances = []
    fold_preds_train = np.array([0.]*len(y))
    fold_preds_oof = np.array([0.]*len(y))
    if return_estimators: fold_estimators = []
    
    
    # 4.1 create list with indexes for train and oof for each split (folds definition)
    data_time=time.time()
    if not fold_column is None:
        splits=[]
        for f in X_Fold.groupby(fold_column)[fold_column].unique():
#             split=[X_Fold[X_Fold[fold_column]!=f[0]].index.values.astype(int), X_Fold[X_Fold[fold_column]==f[0]].index.values.astype(int)]
            split=[]
            split=[X_Fold[X_Fold[fold_column]!=f[0]].index.values.astype(int), X_Fold[X_Fold[fold_column]==f[0]].index.values.astype(int)]
            splits.append(split)
    else:    
        kf = StratifiedKFold(n_splits=nfold, random_state=random_state, shuffle=True) if stratified else KFold(n_splits=nfold, random_state=random_state, shuffle=True)
        splits=kf.split(X, y)
           
    data_time=round(time.time()-data_time, 1)
    
    # 4.2 iterate through folds
    for n, (train_index, oof_index) in enumerate(splits):
        #print('debug. splits:', n, (train_index, oof_index))
        fold_start_time=time.time()
        if verbose!=False and verbose>=0: print('Fold', n+1, 'of', nfold, datetime.datetime.now())
        
        # 4.2.1 create test and valid sets
        X_train, X_oot = X.iloc[train_index], X.iloc[oof_index]
        y_train, y_oot = y.iloc[train_index], y.iloc[oof_index]
        
        # 4.2.2 create and fit estimator
        est = LGBMClassifier_binary(booster_params)
        est.fit(
             X_train, y_train
            ,eval_set=(X_oot, y_oot), eval_names=['oof']            
            ,eval_metric=eval_metric
            ,early_stopping_rounds=early_stopping_rounds            
            ,verbose=verbose
            
            ,feature_name=feature_name
            ,categorical_feature=categorical_feature
        )
        
        # 4.3 gather in fold results
        fold_best_iterations.append(est.best_iteration_)
        fold_best_scores.append(est.best_score_['oof'])
        fold_evals_results.append(est.evals_result_['oof'])
        fold_feature_importances.append(est.feature_importances_)
        fold_preds_train[train_index]=est.predict_proba(X_train)[:,1]
        fold_preds_oof[oof_index]=est.predict_proba(X_oot)[:,1]
        if return_estimators: fold_estimators.append(est)
        fold_train_times.append(time.time()-fold_start_time)
        
    # 5 summarise results
    fold_preds_train=list(fold_preds_train)
    fold_preds_oof=list(fold_preds_oof)
    
    # gather eval metrics in each fold for best round
    fold_best_score={}
    for metric in fold_best_scores[0].keys():
        fold_best_score[metric] = [fold[metric] for fold in fold_best_scores]
    
    
    # get features importance table
    
    # 6 summarise results - one row
    cv_summary = {}
#     cv_summary['eval_all-logloss']=round(log_loss(y, fold_preds_oof),5)
    cv_summary['train_auc']=round(roc_auc_score(y, fold_preds_train), 5)
    cv_summary['oof_auc']=round(roc_auc_score(y, fold_preds_oof), 5)
    cv_summary['oof_auc_range']=round(np.max(fold_best_score[metric])-np.min(fold_best_score[metric]), 5)
#     cv_summary['best_iterations_mean']=int(np.mean(fold_best_iterations))
#     cv_summary['train_time_mean']=round(np.mean(fold_train_times), 1)
#     cv_summary['data_prep_time']=data_time    
    cv_summary['total_time']=round(time.time()-total_time, 1)    
    
#     for metric in fold_best_scores[0].keys():
#         cv_summary['eval_mean-'+metric]=np.mean(fold_best_score[metric])
#     cv_summary['features_used']=str(X.columns)
#     cv_summary['features_importance_mean']=str(list(np.mean(fold_feature_importances, axis=0).astype(int)))
    
    # 7 prep output
    
    return {
        'cv_summary':cv_summary
        ,'train_shape':X.shape
        ,'features_used':list(X.columns)
        ,'features_cat_used':categorical_feature
        ,'features_importance_mean':list(np.mean(fold_feature_importances, axis=0).astype(int))
        ,'fold_best_iteration':fold_best_iterations
        ,'fold_train_time':fold_train_times
        ,'fold_best_score':fold_best_score
        ,'fold_feature_importance':fold_feature_importances
        ,'fold_estimator':fold_estimators
        ,'fold_evals_result':fold_evals_results
        ,'preds':{'train_preds':fold_preds_train,'oof_preds':fold_preds_oof}
    }

    # 4 merge predictions
    # 5 calculate scores (for each fold, mean, score on total oof preds)
    # 6 merge other fold results (best rounds, train time)
    # 7 train single estimator on whole set (with mean best rounds * 1/(1-nfold/2))
 
def LGBMClassifier_binary_cv_grid(
    grid_params={},
    booster_params={}, 
    X=None,
    y=None, 
    feature_name='auto', 
    feature_remove='auto',
    nfold=5, 
    verbose=0,
    random_state=1
):
    # 1 remove grid params from booster params (also check for synonyms)
    for par in lgbm_parameters_synonyms.keys():
        for syn in lgbm_parameters_synonyms[par]:
            if syn in grid_params.keys():
                for syn in lgbm_parameters_synonyms[par]:
                    if syn in booster_params.keys():
                        del booster_params[syn]
    
#     print('grid', grid_params)
#     print('boost', booster_params)
    
    # 2 create placeholders for grid results
    grid_res_lines=[]
    grid_res_details=[]
    
    # 3 create grid
    grid=list(ParameterGrid(grid_params))
    print('Starting grid:', len(grid), 'points.')
    for n, (grid_point_param) in enumerate(grid):
        grid_point_start_time = time.time()
        
        # 3.1 create parameter set for current grid point
        grid_point_params=booster_params.copy()
        grid_point_params.update(grid_point_param)
        
        # 3.2 prep res
        grid_point_res={}
        grid_point_res['grid_point_no']=n+1
        grid_point_res['grid_params_grid']=grid_point_param
        grid_point_res['grid_params_fixed']=booster_params
        
        grid_res_line={}
        grid_res_line['grid_point_no']=n+1
        for par in grid_point_param:
            grid_res_line[par]=grid_point_param[par]
        
        # 3.2 fit cv
        grid_point_res['cv_results']=LGBMClassifier_binary_cv(grid_point_params, X=X, y=y, feature_name=feature_name, feature_remove=feature_remove, nfold=nfold, verbose=verbose, random_state=random_state)
        
#         print(grid_point_res['cv_results'].keys())
#         print(grid_point_res['cv_results']['cv_summary'].keys())
        # 3.3 gather grid point results
        for res in grid_point_res['cv_results']['cv_summary'].keys():
            grid_res_line[res]=grid_point_res['cv_results']['cv_summary'][res]
        
        grid_point_total_time=round(time.time()-grid_point_start_time,1)
        grid_res_line['total_time']=grid_point_total_time
        grid_res_line['total_time']=grid_point_total_time
        grid_res_lines.append(grid_res_line)
        grid_res_details.append(grid_point_res)
        
        # 3.4 print out grid point res        
        print('Grid point', n+1, 'of', len(grid),':', grid_res_line)
                                           
    
    # 4 save results
    return {
        'grid_summary':grid_res_lines
        ,'grid_summary_details':grid_res_details
    }
    
class OrdinalEncoder:
    # Encode feature levels as an ordinal integers array (0 to number of unique feature values - 1).
    #
    # Levels unseen during fit are mapped to highest level
    #
    # 'order': how values are ordered before applying ordinal integers mapping
    # 'num_levels': only top 'num_levels' unique values are mapped to separate integer, all others are mapped to single new integer.
    #               0 - denotes no limit. 
    # 'min_data_in_level', 'min_level_fraction': if > 0 then only levels statisfiying condidion(s) will be mapped to separate integer.
    
    # TODO
    # choose value foe NA
    # choose value for new levels
    # order by weight
    
    from collections import defaultdict
    
    def __init__(self, order='count', num_levels=0, min_data_in_level=0, min_level_fraction=0.0):
        assert order in ['lex', 'count'], "'order' needs to be one of "+str(['lex', 'obs'])
        assert num_levels != 1, "'num_levels' needs to be either 0 (no limit) or > 1"
        
        self.order=order
        self.num_levels=num_levels
        self.min_data_in_level=min_data_in_level
        self.min_level_fraction=min_level_fraction
        
     
    def fit(self, X, feature_name):
        # dictionary for storing values => integer mapping
        self.mappings = defaultdict(int)
        
        #gather unique values and count occurance (sorted descending by count), feature values being frame indexes
        self.levels = X[feature_name].value_counts(dropna=False).to_frame(name='count')
        
        # if order='lex' => sort according to index (feature values lexicographical order)
        if self.order=='lex': self.levels.sort_index(inplace=True)
        
        # drop levels on 'min_data_in_level' and 'min_level_fraction'
        if self.min_data_in_level > 0:
            self.levels = self.levels[self.levels['count'] >= self.min_data_in_level]
        if self.min_level_fraction > 0:
            self.levels['frac'] = self.levels['count']/np.sum(self.levels['count'])
            self.levels = self.levels[self.levels['frac'] >= self.min_level_fraction]
        
        # create new col with row no (new encoding)
        self.levels['encoding']=np.arange(len(self.levels))
        
        # drop levels on 'num_levels'    
        if self.num_levels > 0:
            self.levels = self.levels[self.levels['encoding'] < self.num_levels]
        
        # populate 'mappings': feature values being keys, and ints from 'encoding' values
        self.mappings.update(self.levels.to_dict()['encoding'])
        
        # set default value to max value in mappings
        self.max_index = np.max(list(self.mappings.values()))
        self.mappings.default_factory = lambda: self.max_index
    
    
    def transform(self, X, feature_name, replace=False, suffix='__ordinal'):
        #TODO ?change values of X inplace
        return X[feature_name].map(self.mappings).astype(int)
#         if replace==True:
#             X[feature_name]=X[feature_name].map(mappings)
#         else:
#             X[feature_name+suffix]=X[feature_name].map(mappings)
        
def make_features_contribution_list(X, contrib_dic={}):
    contrib_dic_all_features=OrderedDict()
    
    for c in list(X.columns):
        contrib_dic_all_features[c] = contrib_dic[c]  if c in contrib_dic.keys() else 1.0

    return list(contrib_dic_all_features.values())    
