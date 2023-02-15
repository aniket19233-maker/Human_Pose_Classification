from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from hiclass import LocalClassifierPerNode, LocalClassifierPerLevel, LocalClassifierPerParentNode
import networkx as nx

import pickle
import argparse
from train_utils import *

parser = argparse.ArgumentParser(description='manual to this script')

#model to train
parser.add_argument('--model', default='LR')
parser.add_argument('--feature_type', default='custom')

args = parser.parse_args()

# models
MODELS = {'LR':LogisticRegression(C=1000, max_iter=1000), 'SGD':SGDClassifier(), 'RF':RandomForestClassifier(), 'XGB':XGBClassifier(), 
          'ADA':AdaBoostClassifier(), 'KNN':KNeighborsClassifier(), 'SVM':SVC(), 'GNB':GaussianNB(), 'DT':DecisionTreeClassifier(max_depth=3)}

def draw_graph(classifier):
    nx.draw(classifier.hierarchy, with_labels=True, node_color='Red')

# function to train_models
def run_model(x_train, x_test, y_train, y_test, custom):
    
    model_dict = {}
    ## Local Classifier per Parent Node

    ## create level_wise_labels
    l1_y_train = []
    l2_y_train = []
    l3_y_train = []
    l1_y_test = []
    l2_y_test = []
    l3_y_test = []

    for i in range(len(y_train)):
        l1_y_train.append(y_train[i][2])
        l2_y_train.append(y_train[i][1])
        l3_y_train.append(y_train[i][0])

    for i in range(len(y_test)):
          l1_y_test.append(y_test[i][2])
          l2_y_test.append(y_test[i][1])
          l3_y_test.append(y_test[i][0])

    l1_y_train = pd.Series(l1_y_train)
    l2_y_train = pd.Series(l2_y_train)
    l3_y_train = pd.Series(l3_y_train)

    l1_y_test = pd.Series(l1_y_test)
    l2_y_test = pd.Series(l2_y_test)
    l3_y_test = pd.Series(l3_y_test)
    
    ## train the model
    clf = MODELS[args.model]

    ## LEVEL-1
    l1_clf = clone(clf)
    l1_clf.fit(x_train, l1_y_train)
    l1_y_train_pred = pd.Series(l1_clf.predict(x_train))
    l1_y_test_pred = pd.Series(l1_clf.predict(x_test))
    print("L1 Accuracy:",accuracy_score(l1_y_test_pred,l1_y_test))

    ## LEVEL-2
    l2_clfs = {}
    correct_preds=0
    for pose in set(l1_y_train):
        l2_clfs[pose]=clone(clf)
        train_index = list(l1_y_train_pred[l1_y_train_pred==pose].index)
        test_index = list(l1_y_test_pred[l1_y_test_pred==pose].index)
        
        #test_index = l1_y_test_pred.where(l1_y_test_pred.values==pose)
        temp_x_train = x_train.iloc[train_index]
        temp_y_train = l2_y_train.iloc[train_index]
        temp_x_test = x_test.iloc[test_index]
        temp_y_test = l2_y_test.iloc[test_index]
        
        print(temp_x_test.shape, temp_x_train.shape)

        try:
            l2_clfs[pose].fit(temp_x_train, temp_y_train)
            curr_l2_y_train_pred = l2_clfs[pose].predict(temp_x_train)
            curr_l2_y_test_pred = l2_clfs[pose].predict(temp_x_test)

            correct_preds += (temp_y_test==curr_l2_y_test_pred).sum()
        except:
            pass
    print("L2 Accuracy:",correct_preds/len(y_test))

    ###########
    """
    ## LEVEL-3
    l3_clfs = {}
    correct_preds_l3=0
    for pose in set(l2_train):
        l3_clfs[pose]=clone(clf)

        l3_x_train = []
        l3_y_train = []
        l3_x_test = []
        l3_y_test = []

        for i in range(len(x_train)):
            if l1_y_train_pred[i]==pose:
                l2_x_train.append(x_train.iloc[i,:])
                l2_y_train.append(l2_train[i])

        for i in range(len(x_test)):
            if l1_y_test_pred[i]==pose:
                l2_x_test.append(x_test.iloc[i,:])
                l2_y_test.append(l2_test[i])

        l2_clfs[pose].fit(l2_x_train, l2_y_train)
        l2_y_train_pred = l2_clfs[pose].predict(l2_x_train)
        l2_y_test_pred = l2_clfs[pose].predict(l2_x_test)

        correct_preds += (l2_y_test==l2_y_test_pred).sum()
    print("L2 Accuracy:",correct_preds/len(y_test))
    """
    return model_dict


def get_designed_data_results(df):
    
    x = df.iloc[:,:-3]
    y = [[df.iloc[:,-3][i], df.iloc[:,-2][i], df.iloc[:,-1][i]] for i in range(df.shape[0])]

    x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.25, random_state=0)
    
    model_dict = None
    ## run models
    model_dict = run_model(x_train, x_test, y_train, y_test,True)
    
    return model_dict

#NOT NEEDED
def get_raw_data_results(df):
    
    df["label_3"] = df["level_3"]#level_3.transform(df["level_3"])
    df["label_2"] = df["level_2"]#level_2.transform(df["level_2"])
    df["label_1"] = df["level_1"]#level_1.transform(df["level_1"])
    
    ## dropping Pose column
    df.drop(columns=["level_3","level_2","level_1"],inplace=True)
     
    x = df.iloc[:,:-3]
    y = [[df.iloc[:,-3][i], df.iloc[:,-2][i], df.iloc[:,-1][i]] for i in range(df.shape[0])]

    x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.25, random_state=0)

    model_dict=None
    ## running different models on this
    model_dict = run_model(x_train, x_test, y_train, y_test,False)
    return model_dict

def main():
    
    
    # model_dict = get_raw_data_results(df)
    if args.feature_type == "raw":
        df = pd.read_csv("dataset_hierarchy.csv")
        model_dict = get_raw_data_results(df)
    else:
        df = pd.read_csv("custom_dataset_hierarchy.csv")
        model_dict = get_designed_data_results(df)


if __name__ == '__main__':
    main()
