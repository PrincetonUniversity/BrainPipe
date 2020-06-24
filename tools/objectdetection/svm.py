#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:30:15 2018

@author: tpisano
"""
import numpy as np, pandas as pd, os, sys, matplotlib.pyplot as plt, multiprocessing as mp, time
from tools.utils.io import listdirfull, load_np, makedir, save_kwargs,load_dictionary
from tools.objectdetection.evaluate_performance import pairwise_distance_metrics_multiple_cutoffs
from tools.objectdetection.postprocess_cnn import load_tiff_folder, probabilitymap_to_centers_thresh
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.externals import joblib
from skimage.external import tifffile

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
#TRY THIS WITH RBF nonlinear kernel: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#ALSO TRY NuSVC as well
if __name__ == '__main__':
    from tools.objectdetection.svm import svm_svc
    #inputs
    penalty='l1' # str, ‘l1’ or ‘l2’, default: ‘l2’
    solver='liblinear' #“liblinear”, “newton-cg”, “lbfgs”, “sag” and “saga” #The “saga” solver is often the best choice. The “liblinear” solver is used by default for historical reasons.
    multi_class='ovr' #Multiclass option can be either ‘ovr’ or ‘multinomial’. If the option chosen is ‘ovr’, then a binary problem is fit for each label. Else the loss minimised is the multinomial loss fit across the entire probability distribution. Does not work for liblinear solver.
    max_iter=1000 #100
    dual=False #Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
    tol=0.00001 #Tolerance for stopping criteria
    C=3.0 #1.0 Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    fit_intercept=True
    intercept_scaling=1 #Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight. Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight=None #dict or ‘balanced’, default: None
    random_state=None #The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when solver == ‘sag’ or ‘liblinear’.
    cores = 12 #for parallelization
    kfold_splits = 3 #10 # number of times to iterate through using kfold cross validation
    dst = False#'/home/wanglab/wang/pisano/Python/lightsheet/supp_files/h129_rf_classifier' #place to save classifier
    balance=False
    precision_score=None
    warm_start=True
    verbose=True
    
    #fake data
    tps = np.random.random((50,500))
    fps = np.random.random((50,500))
    
    #train
    logistic_regression(tps, fps, penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, cores = cores, kfold_splits = kfold_splits, dst = dst, balance=balance, precision_score=precision_score, warm_start=warm_start, verbose=verbose)




#%%
def svm_svc(tps, fps, penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, max_iter=100, multi_class='ovr', cores = 10, kfold_splits = 10, dst = False, balance=False, precision_score=None, verbose=True, test_size=0.1):
    '''
    Function to train sklearn's RandomForestClassifier
    Info:
        #http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
        #http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py

    To load and predict:
        rf = joblib.load(pth)
        rf.predict(X_test[0].reshape(1,-1)) <-1,m
        rf.predict(X_test) <-n,m

    Inputs
    ------------
    tps = n,m numpy array of true positivies generated from bin_data
    fps = n,m numpy array of false positivies generated from bin_data
    kfold_splits = number of times to cross validate with sklearn's kfold. Trains on 90%, tests on 10%
    balance = TRAINING ONLY
        False include all data in training
        True make sure the number of training examples for Cell and not Cell are the same. Assumption is more non cells than cells
    class_weight = optional see sklearn.ensemble.randomforest's class_weight
    max_features= optional see sklearn.ensemble.randomforest
    '''
    #verbose = 1 if verbose else 0

    #inputs
    X = np.concatenate((tps,fps), axis=0)
    y = np.asarray([1 for xx in tps] + [0 for xx in fps])
    
    #init
    best_accuracy=0; best_lr = 0; accuracy_lst=[]

    #setup df
    coefficients = pd.DataFrame(data = range(X.shape[1]), columns = ["Feature"])
    accuracy_df = pd.DataFrame(data = None, columns = ["accuracy", 'f1_0', 'f1_1'])
    #KFold cross validation - using straified kfold as it preserves ratios between classes
    if kfold_splits>1:
        gen = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=np.random.seed(10))
    else:
        from sklearn.model_selection import ShuffleSplit
        gen = ShuffleSplit(n_splits = 1, test_size=test_size, random_state=np.random.seed(10))
    for train_index, test_index in gen.split(X, y):
        st = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #optional balance
        if balance:
            ln = len(np.where(y_train == 1)[0])
            idxs = np.concatenate((np.where(y_train == 0)[0][:ln],np.where(y_train == 1)[0][:ln]), axis=0)
            X_train = X_train[idxs]; y_train = y_train[idxs]

        #Fit supervised transformation based on random forests
        from sklearn.metrics import f1_score            
        from sklearn import svm
        lr = svm.SVC
        lr = svm.LinearSVC(penalty=penalty, dual=dual, loss=loss, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, max_iter=max_iter, multi_class=multi_class, verbose=0)
        lr.fit(X_train, y_train)
        #from sklearn.linear_model import SGDClassifier
        #lr = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=50, shuffle=True, verbose=0, epsilon=0.1, n_jobs=cores, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=True, average=True)
        #lr.fit(X_train, y_train)
        
        
        #Performance
        accuracy = lr.score(X_test, y_test); accuracy_lst.append(accuracy)
        f1 = f1_score(y_test, lr.predict(X_test), average=precision_score)
        print ('Classifier accuracy: {}, f1: {} in {} min\n   y_train: {}\n   y_test: {}\n'.format(accuracy, f1, round((time.time() - st)/60, 2), zip(*np.unique(y_train, return_counts=True)), zip(*np.unique(y_test, return_counts=True))))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
            x_train_to_use_for_roc = X_train
            x_test_to_use_for_roc = X_test
            y_train_to_use_for_roc = y_train
            y_test_to_use_for_roc = y_test
        coefficients['{}'.format(len(coefficients.columns))] = np.squeeze(np.transpose(lr.coef_))
        accuracy_df.loc[len(accuracy_df)+1] = [accuracy, f1[0], f1[1]]
        #coefficients = pd.DataFrame({"Feature":X.columns,"Coefficients":np.transpose(lr.coef_)})
        #coefficients = pd.DataFrame({"Feature":range(X.shape[1]),"Coefficients":np.squeeze(np.transpose(lr.coef_))})
    
    out = {'true_positives': tps, 'false_positives': fps, 'X_train': x_train_to_use_for_roc, 'X_test': x_test_to_use_for_roc,
           'y_train': y_train_to_use_for_roc, 'y_test': y_test_to_use_for_roc, 'classifier': best_lr, 'accuracy': np.mean(accuracy_lst), 
           'coefficients':coefficients, 'accuracy_df':accuracy_df}

    #save out
    if dst:
        if dst[-4:] != '.pkl': dst = dst+'.pkl'
        joblib.dump(best_lr, dst)
        out['dst'] = dst

    return out