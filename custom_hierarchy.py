from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from hiclass import LocalClassifierPerNode, LocalClassifierPerLevel, LocalClassifierPerParentNode
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import argparse
from train_utils import *

parser = argparse.ArgumentParser(description='manual to this script')

#model to train
parser.add_argument('--model', default='LR')
parser.add_argument('--feature_type', default='custom')

args = parser.parse_args()

## DT with entropy is the best
# models
MODELS = {'LR':LogisticRegression(C=1000, max_iter=1000), 'SGD':SGDClassifier(), 'RF':RandomForestClassifier(), 'XGB':XGBClassifier(), 
          'ADA':AdaBoostClassifier(), 'KNN':KNeighborsClassifier(), 'SVM':SVC(), 'GNB':GaussianNB(), 'DT':DecisionTreeClassifier(max_depth=5, criterion="entropy")}

FEATURE_NAMES=None


def draw_graph(classifier):
    nx.draw(classifier.hierarchy, with_labels=True, node_color='Red')

# function to train_models
def run_model(x_train, x_test, y_train, y_test, custom):
    
    l1_acc=None
    l2_acc=None
    l3_acc=None

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
    l1_acc=accuracy_score(l1_y_test_pred,l1_y_test)
    print("L1 Accuracy:",l1_acc)

    ### print
#     fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (16,16), dpi=1200)
#     plot_tree(l1_clf,
#                feature_names = FEATURE_NAMES, 
#                class_names=list(set(l1_y_train)),
#                filled = True)
    fig.savefig('imagename.png')
    ###

    ## LEVEL-2
    l2_clfs = {}
    correct_preds=0
    total=0
    l2_y_train_pred=[]
    l2_y_test_pred=[]
    for pose in set(l1_y_train).union(set(l1_y_test)):
        l2_clfs[pose]=clone(clf)
        train_index = list(l1_y_train_pred[l1_y_train_pred==pose].index)
        test_index = list(l1_y_test_pred[l1_y_test_pred==pose].index)
        
        #test_index = l1_y_test_pred.where(l1_y_test_pred.values==pose)
        temp_x_train = x_train.iloc[train_index]
        temp_y_train = l2_y_train.iloc[train_index]
        temp_x_test = x_test.iloc[test_index]
        temp_y_test = l2_y_test.iloc[test_index]
        
     #   print(set(temp_y_test))

        try:
            l2_clfs[pose].fit(temp_x_train, temp_y_train)
            curr_l2_y_train_pred = l2_clfs[pose].predict(temp_x_train)
            curr_l2_y_test_pred = l2_clfs[pose].predict(temp_x_test)

            l2_y_train_pred.extend(curr_l2_y_train_pred)
            l2_y_test_pred.extend(curr_l2_y_test_pred)

            correct_preds += (temp_y_test==curr_l2_y_test_pred).sum()
            total += len(temp_y_test)
        except:
            pass
    l2_acc = l1_acc*correct_preds/total
    print("L2 Accuracy:",l2_acc)

    l2_y_train_pred=pd.Series(l2_y_train_pred)
    l2_y_test_pred=pd.Series(l2_y_test_pred)
    
    ###########
    ## LEVEL-3
    l3_clfs = {}
    correct_preds=0
    total=0
    for pose in set(l2_y_train).union(set(l2_y_test)):
        l3_clfs[pose]=clone(clf)
        train_index = list(l2_y_train_pred[l2_y_train_pred==pose].index)
        test_index = list(l2_y_test_pred[l2_y_test_pred==pose].index)
        
        #test_index = l1_y_test_pred.where(l1_y_test_pred.values==pose)
        temp_x_train = x_train.iloc[train_index]
        temp_y_train = l3_y_train.iloc[train_index]
        temp_x_test = x_test.iloc[test_index]
        temp_y_test = l3_y_test.iloc[test_index]
        
        #print(set(temp_y_test))

        try:
            l3_clfs[pose].fit(temp_x_train, temp_y_train)
            curr_l3_y_train_pred = l3_clfs[pose].predict(temp_x_train)
            curr_l3_y_test_pred = l3_clfs[pose].predict(temp_x_test)

            correct_preds += (temp_y_test==curr_l3_y_test_pred).sum()
            total += len(temp_y_test)
        except:
            pass
    l3_acc=l2_acc*correct_preds/total
    print("L3 Accuracy:",l3_acc)
    return model_dict


def get_designed_data_results(df):
    
    FEATURE_NAMES=df.columns[1:-3].values
    x = df.iloc[:,1:-3]
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
     
    x = df.iloc[:,1:-3]
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
