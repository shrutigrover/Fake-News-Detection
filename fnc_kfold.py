from __future__ import print_function
import os
import sys
import numpy as np
import json
import pandas as pd
import time

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, NMF_cos_50, LDA_cos_25
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

#Model 2 dependencies
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import matplotlib.pyplot as plt

train_feature_data =  pd.DataFrame(columns=['headline','body_id','stance'])
comp_feature_data =  pd.DataFrame(columns=['headline','body_id','stance'])
def generate_features(stances,dataset,name):
    h, b, y = [],[],[]
    rows = []
    for stance in stances:
        row = []
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
        row.append(stance['Headline'])
        row.append(dataset.articles[stance['Body ID']])
        row.append(LABELS.index(stance['Stance']))
        rows.append(row)

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    ######Topic Modelling - New Features Added######
    X_NMF = gen_or_load_feats(NMF_cos_50, h, b, "features/nmf."+name+".npy")
    X_LDA = gen_or_load_feats(LDA_cos_25, h, b, "features/lda-25."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_NMF, X_LDA]
    if(name == "competition"):
        if not (os.path.isfile('comp_feature_data.csv')):
            comp_feature_data['stance'] = y
            comp_feature_data['headline'] = h
            comp_feature_data['body_id'] = b
            for i in range(0,X.shape[1]):
                comp_feature_data[i] = X[:,i]

    if(name == "full"):
        if not (os.path.isfile('train_feature_data.csv')):
            train_feature_data['stance'] = y
            train_feature_data['headline'] = h
            train_feature_data['body_id'] = b
            for i in range(0,X.shape[1]):
                train_feature_data[i] = X[:,i]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()

    X_full,y_full = generate_features(d.stances,d,"full")
    #for binary classification - related and unrelated
    y_full = [x if x==3 else 2 for x in y_full]

    #removing folds return train and holdout split - check if distribution same - does it matter
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    y_holdout = [x if x==3 else 2 for x in y_holdout]

    #load training data
    X_train, y_train = generate_features(fold_stances, d, "train_n")
    y_train = [x if x==3 else 2 for x in y_train]

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
    y_competition = [x if x==3 else 2 for x in y_competition]

    #param = {'eta':1, 'objective' : 'multi:softmax' , 'num_class' : 4, 'n_estimators':150}
    param = {'eta':1, 'objectve' : "binary:logistic" , 'n_estimators':150, 'seed':10}

    clf = XGBClassifier(**param)
    start = int(round(time.time()*1000))
    end = int(round(time.time()*1000))
    train_time = end - start
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_holdout, y_holdout)], verbose=True)

    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_holdout)
    y_pred_onfull = clf.predict(X_full)

    if not (os.path.isfile('train_feature_data.csv')):
        train_feature_data['predicted_stance'] = y_pred_onfull
        train_feature_data.to_csv('train_feature_data.csv', index = False)
        #check file
        feature_df = pd.read_csv('train_feature_data.csv')
        print("train data file size : ", feature_df.shape)
        print("train data file: ", feature_df.head())

    predicted = [LABELS[int(a)] for a in y_pred_train]
    actual = [LABELS[int(a)] for a in y_train]
    print("Scores on the train set")
    report_score(actual,predicted)
    print("")
    print("")

    predicted = [LABELS[int(a)] for a in y_pred]
    actual = [LABELS[int(a)] for a in y_holdout]
    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    test_pred = clf.predict(X_competition)
    predicted = [LABELS[int(a)] for a in test_pred]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)

    if not (os.path.isfile('comp_feature_data.csv')):
        comp_feature_data['predicted_stance'] = test_pred
        comp_feature_data.to_csv('comp_feature_data.csv', index = False)
        #check file
        feature_df = pd.read_csv('comp_feature_data.csv')
        print("comp data file size : ", feature_df.shape)
        print("comp data file: ", feature_df.head())

    print("train time: ",train_time)
