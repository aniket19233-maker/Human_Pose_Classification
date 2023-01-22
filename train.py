from imblearn.over_sampling import SMOTE
import sklearn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import mediapipe as mp
import cv2
import os
from xgboost import XGBClassifier
from tqdm import tqdm

import pickle
import copy
import argparse
from train_utils import *

parser = argparse.ArgumentParser(description='manual to this script')

#model to train
parser.add_argument('--model', default='LR')

args = parser.parse_args()

# models
MODELS = {'LR':LogisticRegression(), 'SGD':SGDClassifier(), 'RF':RandomForestClassifier(), 'XGB':XGBClassifier(), 'ADA':AdaBoostClassifier(), 'KNN':KNeighborsClassifier(), 'SVM':SVC()}
GRID_SRCH_PARAMS = {
    'LR':{'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1]}, 
    'SGD':{'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 'max_iter': [100,500,1000], 
          'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 
          'penalty': ['l1', 'l2', 'elasticnet'], 
          'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']},
    'RF':{'bootstrap': [True], 'max_depth': [5, 10, 50, 100], 
          'max_features': ['auto', 'log2'], 'n_estimators': [5, 10, 50, 100]},
    'KNN':{'metric':['euclidean','manhattan'] ,'n_neighbors': np.arange(1, 16),
          'algorithm':{'auto', 'ball_tree', 'kd_tree', 'brute'}},     
    'ADA':{'n_estimators':[5, 10, 50, 100], 
          'learning_rate':[0.0001, 0.001, 0.01, 0.1, 1.0]}, 
    }

# function to train_models
def run_model(x_train, x_test, y_train, y_test, custom):
    
    model_dict = {}
    
    ## train the model
    model = MODELS[args.model]
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(25*"##")
    print(args.model)
    print("Testing Data:")
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("F1 Score:",f1_score(y_test,y_pred, average="weighted"))
    #print(classification_report(y_test,y_pred))
    #print(25*"**")
    y_pred = model.predict(x_train)
    print(25*"--")
    print("Training Data:")
    print("Accuracy:",accuracy_score(y_train,y_pred))
    print("F1 Score:",f1_score(y_train,y_pred, average="weighted"))
    ##print(classification_report(y_train,y_pred))
    print(25*"##")
    print("USING HYPERTUNING")
    grid_vals = GRID_SRCH_PARAMS[args.model]
    grid_lr = GridSearchCV(estimator=model, param_grid=grid_vals, 
    scoring='accuracy', cv=5, refit=True, return_train_score=True) 

    #Training and Prediction

    grid_lr.fit(x_train, y_train)
    print(grid_lr.best_estimator_)
    preds = grid_lr.best_estimator_.predict(x_test)
    print("Testing Data:")
    print("Accuracy:",accuracy_score(y_test,preds))
    print("F1 Score:",f1_score(y_test,preds, average="weighted"))
    
    model_dict[args.model] = model
    name = "raw_feature_model_dict"
    if custom:
        name = "custom_feature_model_dict.pkl"
    
    with open(name,"wb") as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return model_dict


def get_designed_data_results(df):
    
    ## converting output to numeric values
    le.fit(df["Pose"])
    df["label"] = le.transform(df["Pose"])
    
    ## dropping Pose column
    df.drop(columns=["Pose","ImgNum"],inplace=True)
    
    ## feature designing for train and test
    df = get_designed_data_df(df)
    
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.25, random_state=0)
    
    model_dict = None
    ## run models
    model_dict = run_model(x_train, x_test, y_train, y_test,True)
    
    return model_dict

def main():
    
    df = pd.read_csv("trainSet_yoga82.csv")
    # model_dict = get_raw_data_results(df)
    d = get_designed_data_results(df)


if __name__ == '__main__':
    main()
