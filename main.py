import generateFeatures
import HPO

import scipy.stats as ss
import sklearn.model_selection as mls
import lightgbm as lgb
import os
import pandas as pd
import statistics
import pickle

def create_model(modelFile, metric = 'mae', num_iterations = 100, labels = '', selectfeature_params = (False, None), HPO_params = (False, {'boosting': 'gbdt', 'metric': {'mae'}, 'num_leaves': 31}), saveResults=True, saveFeatures=True):
    """create predictive using the lightgbm algorithm and a pandas dataframe"""
    # Prepare train data.
    if os.path.isfile('trainDataPrepared.csv') == False:
        #process dataset
        print("Processing train data...")
        trainDataProc, relfeatures = generateFeatures.generate_features(trainDataLocation, 'train', saveResults = saveResults, saveFeatures = saveFeatures)
    elif os.path.isfile('trainDataPrepared.csv'):
        # Load in existing processed dataset.
        print("Loading train data...")
        trainDataProc = pd.read_csv('trainDataPrepared.csv', header=0)
        relfeatures = pickle.load(open('features.pkl', 'rb'))

    #select features based on feature importances
    if selectfeature_params[0] and selectfeature_params[1] == 'feature_importance':
        #select features based on feature importance in model
        if os.path.isfile(f'selectedfeatures_{selectfeature_params[2]}featimportance.pkl') == False:
            print ('Selecting features using feature importances')
            relfeatures = select_feature_importance(selectfeature_params[2],
                                                     impType = selectfeature_params[3],
                                                     saveSelFeatures = True,
                                                     labels = labels)
        elif os.path.isfile(f'selectedfeatures_{selectfeature_params[2]}featimportance.pkl'):
            print ('Loading selected features using feature importances')
            relfeatures = pickle.load(open(f'selectedfeatures_{selectfeature_params[2]}featimportance.pkl', 'rb'))
    #TODO: create function in featureSelection for Anova test and apply it in main

    #Prepare train X and Y column names.
    trainColumnsX = relfeatures
    trainColumnsY = labels

    #perform hyperparameter optimization
    if HPO_params[0] and os.path.isfile(f'{HPO_params[2]}_params.pkl') == False:
        print ('Performing hyperparameter optimization')
        params = HPO.HPO(trainDataProc,
                         HPO_params,
                         features = trainColumnsX,
                         labels = trainColumnsY,
                         metric = metric,
                         num_iterations = num_iterations,
                         saveResults = True,
                         saveParams = True)
    elif HPO_params[0] and os.path.isfile(f'{HPO_params[2]}_params.pkl'):
        print ('Loading optimized parameters')
        params = pickle.load(open(f'{HPO_params[2]}_params.pkl', 'rb'))

    #choose specified params if HPO not wanted
    else:
        params = HPO_params[1]

    # Perform K-fold split and prepare model.
    kfold = mls.KFold(n_splits=5, shuffle=True, random_state=0)
    result = next(kfold.split(trainDataProc), None)
    train = trainDataProc.iloc[result[0]]
    test = trainDataProc.iloc[result[1]]

    # Train model via lightGBM.
    lgbTrain = lgb.Dataset(train[trainColumnsX], train[trainColumnsY])
    lgbEval = lgb.Dataset(test[trainColumnsX], test[trainColumnsY])

    # Set up training.
    print("Beginning training")
    gbm = lgb.train(params,
                    lgbTrain,
                    num_boost_round=200,
                    valid_sets=lgbEval,
                    early_stopping_rounds=200)

    #save model
    print("Saving model")
    gbm.save_model(modelFile)
    print("Model saved.")

    return gbm

def create_predictions(modelFile, labels, selectfeature_params = [False, None], saveResults = True, saveFeatures = False):
    """generate predictions based on model created in create_model()"""
    #Prepare test data.
    if os.path.isfile('testDataPrepared.csv') == False:
        # Process test data
        print("Processing test data...")
        testDataProc, relfeatures = generateFeatures.generate_features(testDataLocation, 'test', saveResults = saveResults, saveFeatures = saveFeatures)
    elif os.path.isfile('testDataPrepared.csv'):
        # Load in existing processed dataset.
        print("Loading test data...")
        testDataProc = pd.read_csv('testDataPrepared.csv', header=0)
        relfeatures = pickle.load(open('features.pkl', 'rb'))

    #select features
    if selectfeature_params[0] and selectfeature_params[1] == 'feature_importance':
        #select features based on feature importance in model
        if os.path.isfile(f'selectedfeatures_{selectfeature_params[2]}featimportance.pkl') == False:
            print ('Selecting features using feature importances')
            relfeatures = select_feature_importance(selectfeature_params[2],
                                                     impType = selectfeature_params[3],
                                                     saveSelFeatures = True,
                                                     labels = labels)
        elif os.path.isfile(f'selectedfeatures_{selectfeature_params[2]}featimportance.pkl'):
            print ('Loading selected features using feature importances')
            relfeatures = pickle.load(open(f'selectedfeatures_{selectfeature_params[2]}featimportance.pkl', 'rb'))
    #TODO: create function in for Anova test and apply it in main

    #Prepare train X and Y column names.
    testSubmissionX = relfeatures

    # generate predictions
    print("Generating predictions.")
    currentModel = lgb.Booster(model_file=modelFile)
    prediction = currentModel.predict(testDataProc[testSubmissionX])
    print(statistics.mean(prediction))
    testDataProc[labels[0]] = prediction

    return testDataProc, prediction

def select_feature_importance(threshold, impType = 'gain', saveSelFeatures = True, labels = ''):
    """select features using feature importances in initial model"""
    allfeatures = pickle.load(open('features.pkl', 'rb'))

    #ensures that initial model exists. If model containing all features does not exist, will create one immediately. Beware of memory error.
    if os.path.isfile('model_allfeatures.txt') == False: #VERY DANGEROUS. DO NOT RECOMMEND
        print ('Initial model not found. Creating model with all features now')
        create_model(modelFile = 'model_allfeatures.txt', labels = labels)

    #create dataframe containing all features and their feature importances
    model = lgb.Booster(model_file = 'model_allfeatures.txt')
    feat_importances = pd.DataFrame(model.feature_importance(importance_type = impType), columns = ['importance'])
    feat_importances['feat_name'] = allfeatures

    #create list containing all features which have feature importances above the threshold. The first element is the threshold which is used for naming the model in main.py
    selectedfeatures = [feat_importances['feat_name'][i] for i in range(len(feat_importances)) if feat_importances['importance'][i]>=threshold]
    print ('Selected features are:', selectedfeatures)

    #save selected features into pickle file
    if saveSelFeatures:
        print (f'Saving selected features for {threshold} threshold')
        with open(f'selectedfeatures_{threshold}featimportance.pkl', 'wb') as f:
            pickle.dump(selectedfeatures, f)

    return selectedfeatures

######################################################################################################################
# Variables
trainDataLocation = "train.csv"
testDataLocation = "test.csv"
modelFile = 'model_10000features.txt'
metric = 'mae'
num_iterations = 1000
labels = ['scalar_coupling_constant']
selfeat_params = (True, 'feature_importance', 10000, 'gain')
params_range= {'learning_rate': ss.uniform(0.01, 2.0),
                 'min_child_samples': ss.randint(1, 100),
                 'num_leaves': ss.randint(5, 500),
                 'feature_fraction': ss.uniform(0.01, 0.9),
                 'bagging_fraction': ss.uniform(0.01, 0.9),
                 'bagging_freq': ss.randint(5, 50),
                 'reg_alpha': [0, 1],
                 'reg_lambda': [0, 1],
                 'subsample': ss.uniform(0.2, 0.7)}
hyperparam_opt = (True, 'regression', 'random', params_range, 20)

#####################################################################################################################
#Create model or create predictions
if os.path.exists(modelFile) == False:
    model = create_model(modelFile, metric, num_iterations, labels, selectfeature_params = selfeat_params, HPO_params = hyperparam_opt)
    print ('Please run again to generate predictions on test data')
else:
    print ('Model already exists in root directory, loading these up')
    testDataProc, prediction = create_predictions(modelFile, labels, selectfeature_params = selfeat_params)
    print ('test data:')
    print (testDataProc.head(5))
    print ('prediction:')
    print (prediction)
