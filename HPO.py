import pandas as pd
import pickle
import numpy as np
import time
import lightgbm as lgb
import sklearn.model_selection as mls


def HPO(trainData, HPO_params, features, labels, metric = 'mae', num_iterations = 1000, saveResults = True, saveParams = True):
    """perform HPO"""
    #begin timing
    start=time.time()

    #begin constructing model
    print ('Constructing Model')
    #initial set of parameters
    params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'num_leaves': 20,
              'learning_rate': 0.05,
              'feature_fraction': 0.5,
              'bagging_fraction': 0.5,
              'reg_alpha': 5,
              'reg_lambda': 10,
              'subsample': 0.5,
              'metric' : 'l1'}
    #construct gradient boosting model
    if HPO_params[1] == 'regression':
        mdl = lgb.LGBMRegressor(boosting_type= 'gbdt',
                              objective = 'regression',
                              silent = True,
                              num_leaves = params['num_leaves'],
                              learning_rate = params['learning_rate'],
                              feature_fraction = params['feature_fraction'],
                              bagging_fraction = params['bagging_fraction'],
                              reg_alpha = params['reg_alpha'],
                              reg_lambda= params['reg_lambda'],
                              subsample= params['subsample'],
                              metric=params['metric'])
    elif HPO_params[1] == 'classifier':
        mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                                objective='multiclass',
                                silent=True,
                                num_leaves=params['num_leaves'],
                                learning_rate=params['learning_rate'],
                                feature_fraction=params['feature_fraction'],
                                bagging_fraction=params['bagging_fraction'],
                                reg_alpha=params['reg_alpha'],
                                reg_lambda=params['reg_lambda'],
                                subsample=params['subsample'],
                                metric=params['metric'])

    #perform HPO
    print ('Begin HPO')
    if HPO_params[2] == 'random':
        randParams = HPO_params[3]
        HPO_results = randsearch(trainData, features, labels, mdl, randParams, num_iter = HPO_params[4], saveResults = saveResults)
    elif HPO_params[2] == 'grid':
        gridParams = HPO_params[3]
        HPO_results = gridsearch(trainData, features, labels, mdl, gridParams, saveResults = saveResults)

    HPO_results['boosting_type'] = 'gbdt'
    if HPO_params[1] == 'regression':
        HPO_results['objective'] = 'regression'
    elif HPO_params[1] == 'classifier':
        HPO_results['objective'] = 'multiclass'
    HPO_results['metric'] = {metric}
    HPO_results['num_iterations'] = num_iterations
    HPO_results['verbose'] = 0

    if saveParams:
        with open(f'{HPO_params[2]}_params.pkl', 'wb') as f:
            pickle.dump(HPO_results, f)


    end = time.time()
    print('time =', (end - start) / 60)

    return HPO_results

def randsearch(trainData, features, labels, mdl, randParams, num_iter, saveResults = True):
    """perform random search"""
    #split data into training and test sets
    modelfeatures = trainData[features]
    modellabels = np.array(trainData[labels]).reshape((-1,))
    train_features, test_features, train_labels, test_labels = mls.train_test_split(modelfeatures, modellabels, test_size=0.2,
                                                                                    random_state=50)

    #select random sets of hyperparameters from a very large grid
    grid = mls.RandomizedSearchCV(mdl, randParams, verbose=0, n_iter=num_iter, cv=5, n_jobs=1)

    #apply LGBMRegressor to randomly picked sets of hyperparameters
    grid.fit(train_features, train_labels)

    #save results as dataframe
    if saveResults:
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv('randHPO_results.csv')

    best_params = grid.best_params_

    return best_params

def gridsearch(trainData, features, labels, mdl, gridParams, saveResults = True):
    """perform grid search"""
    #split data into training and test sets
    modelfeatures = trainData[features]
    modellabels = np.array(trainData[labels]).reshape((-1,))
    train_features, test_features, train_labels, test_labels = mls.train_test_split(modelfeatures, modellabels, test_size=0.2,
                                                                                    random_state=50)
    #create hyperparameter grid
    grid = mls.GridSearchCV(mdl, gridParams, verbose=0, cv=5, n_jobs=1)

    #apply LGBMRegressor to all sets of hyperparameters in grid
    grid.fit(train_features, train_labels)

    #save results as dataframe
    if saveResults:
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv('gridHPO_results.csv')

    best_params = grid.best_params_

    return best_params
