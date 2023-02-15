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
          'ADA':AdaBoostClassifier(), 'KNN':KNeighborsClassifier(), 'SVM':SVC(), 'GNB':GaussianNB()}

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
    print(y_pred[:5])
    draw_graph(lcpn)
    return None
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
    
    model_dict[args.model] = model
    name = "raw_feature_model_dict"
    if custom:
        name = "custom_feature_model_dict.pkl"
    
    with open(name,"wb") as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return model_dict


def get_designed_data_results(df):
    
    ## converting output to numeric values
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
    
    x = df.iloc[:,:-3]
    y = [df.iloc[:,-3],df.iloc[:,-2],df.iloc[:,-1]]
    
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
    y = [df.iloc[:,-3],df.iloc[:,-2],df.iloc[:,-1]]
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.25, random_state=0)

    model_dict=None
    ## running different models on this
    model_dict = run_model(x_train, x_test, y_train, y_test,False)
    return model_dict

def main():
    
    df = pd.read_csv("dataset_hierarchy.csv")
    # model_dict = get_raw_data_results(df)
    if args.feature_type == "raw":
        model_dict = get_raw_data_results(df)
    else:
        model_dict = get_designed_data_results(df)


if __name__ == '__main__':
    main()
