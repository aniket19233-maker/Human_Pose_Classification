from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

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
from hiclass import LocalClassifierPerNode
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
          'ADA':AdaBoostClassifier(), 'KNN':KNeighborsClassifier(), 'SVM':SVC(), 'GNB':GaussianNB(), 'DT':DecisionTreeClassifier(max_depth=1)}

def draw_graph(classifier):
    nx.draw(classifier.hierarchy, with_labels=True, node_color='Red')

# function to train_models
def run_model(x_train, x_test, y_train, y_test, custom):
    
    model_dict = {}
    
    ## train the model
    clf = MODELS[args.model]
    model = LocalClassifierPerNode(local_classifier=clf)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(y_pred[0])
    ## create l1, l2, l3
    l1_pred = []
    l2_pred = []
    l3_pred = []
    l1_test = []
    l2_test = []
    l3_test = []
    
    for i in range(len(y_pred)):
          l1_pred.append(eval(y_pred[i][0]))
          l2_pred.append(eval(y_pred[i][1]))
          l3_pred.append(eval(y_pred[i][2]))
          l1_test.append(y_test[i][0])
          l2_test.append(y_test[i][1])
          l3_test.append(y_test[i][2])
          
    print(l1_pred)
    print(l1_test)
    print(25*"##")
    print(args.model)
    print("Testing Data:")
    print("L1 Accuracy:",accuracy_score(l1_pred,l1_test))
    print("L2 Accuracy:",accuracy_score(l2_pred,l2_test))
    print("L3 Accuracy:",accuracy_score(l3_pred,l3_test))
        
    model_dict[args.model] = model
    name = "raw_feature_model_dict"
    if custom:
        name = "custom_feature_model_dict.pkl"
    
    with open(name,"wb") as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return model_dict


def get_designed_data_results(df):
    
    ## converting output to numeric values
    """
    level_3.fit(df["level_3"])
    level_2.fit(df["level_2"])
    level_1.fit(df["level_1"])

    df["label_3"] = level_3.transform(df["level_3"])
    df["label_2"] = level_2.transform(df["level_2"])
    df["label_1"] = level_1.transform(df["level_1"])
    
    ## dropping Pose column
    df.drop(columns=["level_3","level_2","level_1"],inplace=True)
    
    ## feature designing for train and test
    df = get_designed_data_df(df)
    """
    x = df.iloc[:,:-3]
    y = [[level_3.inverse_transform(df.iloc[:,-3][i]), level_2.inverse_transform(df.iloc[:,-2][i]), level_1.inverse_transform(df.iloc[:,-1][i])] for i in range(df.shape[0])]

    x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.25, random_state=0)
    
    model_dict = None
    ## run models
    model_dict = run_model(x_train, x_test, y_train, y_test,True)
    
    return model_dict

#NOT NEEDED
def get_raw_data_results(df):
    
   ## converting output to numeric values
    level_3.fit(df["level_3"])
    level_2.fit(df["level_2"])
    level_1.fit(df["level_1"])

    df["label_3"] = level_3.transform(df["level_3"])
    df["label_2"] = level_2.transform(df["level_2"])
    df["label_1"] = level_1.transform(df["level_1"])
    
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
    
    dummy = pd.read_csv("dataset_hierarchy.csv")
    level_3.fit(dummy["level_3"])
    level_2.fit(dummy["level_2"])
    level_1.fit(dummy["level_1"]
    # model_dict = get_raw_data_results(df)
    if args.feature_type == "raw":
        df = pd.read_csv("dataset_hierarchy.csv")
        model_dict = get_raw_data_results(df)
    else:
        df = pd.read_csv("custom_dataset_hierarchy.csv")
        model_dict = get_designed_data_results(df)


if __name__ == '__main__':
    main()
