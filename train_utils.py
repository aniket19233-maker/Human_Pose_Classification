#SCORING
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

import mediapipe as mp
import cv2
import os
from xgboost import XGBClassifier
from tqdm import tqdm

import pickle
import copy
import argparse
from train_utils import *

le = preprocessing.LabelEncoder()

def get_cosine_similarity_results(x_train, x_test, y_train, y_test, features):
    #print(x_train[:10])
    #print(y_train[:10])
    y_pred = []
    
    dataset = pd.concat([x_train, y_train], axis=1)
    dict1 = {} #pose -> 132 size array with mean keypoints

    for pose in range(0,81):
      temp = dataset.loc[dataset['label'] == pose].mean()
      temp = temp.drop(labels=['label'])
      dict1[pose] = temp
    #print(dict1)
    
    with open("cosine_dict.pkl","wb") as handle:
        pickle.dump(dict1,handle)
        
    feature_wise_cosine_similarity = {} ## i'th data_point -> {}
    
    feature_wise_cosine_similarity["data_point"] = []
    feature_wise_cosine_similarity["actual_pose"] = []
    feature_wise_cosine_similarity["predicted_pose"] = []
    
    for i in tqdm(range(x_test.shape[0])):

        curr_row = [x_test.iloc[i]]

        curr_y = -1
        curr_sim_score = -float("inf")

        for pose in range(0,81):
          row = dict1[pose]
          #print(row)
          #print("----")
          #print(curr_row)
          row = row.to_numpy()
          curr_row = np.array(curr_row)
          sim_score = cosine_similarity(row.reshape(1, -1),curr_row.reshape(1,-1))

          if sim_score > curr_sim_score:
            curr_sim_score = sim_score
            curr_y = pose

        """for j in range(x_train.shape[0]):

            row = [x_train.iloc[j,:]]

            sim_score = cosine_similarity(row,curr_row)

            if sim_score > curr_sim_score:
                curr_sim_score = sim_score
                curr_y = y_train.iloc[j]"""
        
        feature_wise_cosine_similarity["data_point"].append(i)
        feature_wise_cosine_similarity["predicted_pose"].append(le.inverse_transform([curr_y])[0])
        feature_wise_cosine_similarity["actual_pose"].append(le.inverse_transform([list(y_test)[i]])[0])
        
        row = dict1[curr_y]
        for j in range(len(features)-1):
            
            if features[j] not in feature_wise_cosine_similarity:
                feature_wise_cosine_similarity[features[j]] = []
                
            n1 = row[j]
            n2 = curr_row[0][j]
            sim = 1 - abs(n1 - n2) / (n1 + n2)
            feature_wise_cosine_similarity[features[j]] = sim
        
        y_pred.append(curr_y)

    y_test_list = list(y_test)
    
    print(25*"##")
    print("COSINE SIMILARITY RESULTS:")
    print("Accuracy:",accuracy_score(y_test_list,y_pred))
    print("F1 Score:",f1_score(y_test_list,y_pred, average="weighted"))
    
    correction_df = pd.DataFrame(feature_wise_cosine_similarity)
    
    if len(features) >= 100:
        correction_df.to_csv("correction_dataframe_raw.csv")
    else:
        correction_df.to_csv("correction_dataframe_custom.csv")
    
    print(25*"##")
    
def cosine_eval(x_test, y_test, y_pred, features):
    #print(x_train[:10])
    #print(y_train[:10])
    
    with open("cosine_dict.pkl","rb") as handle:
        dict1 = pickle.load(handle)
        
    feature_wise_cosine_similarity = {} ## i'th data_point -> {}
    
    feature_wise_cosine_similarity["data_point"] = []
    feature_wise_cosine_similarity["actual_pose"] = []
    feature_wise_cosine_similarity["predicted_pose"] = []
    
    for i in tqdm(range(x_test.shape[0])):

        curr_row = [x_test.iloc[i]]
      
        feature_wise_cosine_similarity["data_point"].append(i)
        
        feature_wise_cosine_similarity["predicted_pose"].append(le.inverse_transform([y_pred[i]])[0])
        feature_wise_cosine_similarity["actual_pose"].append(le.inverse_transform([list(y_test)[i]])[0])
        
        row = dict1[y_pred[i]]
        for j in range(len(features)-1):
            
            if features[j] not in feature_wise_cosine_similarity:
                feature_wise_cosine_similarity[features[j]] = []
                
            n1 = row[j]
            n2 = curr_row[0][j]
            sim = 1 - abs(n1 - n2) / (n1 + n2)
            feature_wise_cosine_similarity[features[j]] = sim
        

    y_test_list = list(y_test)
    
    print("dataframe saved with eval results!!")
    
    correction_df = pd.DataFrame(feature_wise_cosine_similarity)
    featurecopy = copy.deepcopy(feature_wise_cosine_similarity)
    del featurecopy['data_point']
    del featurecopy['actual_pose']
    del featurecopy['predicted_pose']
    featurecopy = sorted(featurecopy.items(), key=lambda x: x[1])
    #print(featurecopy[0][0])
    # dealing with predicted_pose key
    # print : top 3 keypoints with least similarity score
    print(25*"##")
    print("Top 3 keypoints to be corrected : ")
    print()
    count = 0
    for k in range(0, len(featurecopy)):
        
        if count>2:
            break
    
        if "centroid" not in featurecopy[k][0]:
            print(featurecopy[k][0], ":", featurecopy[k][1])
            count += 1
    print(25*'##')
    
    if len(features) >= 100:
        correction_df.to_csv("correction_dataframe_raw.csv")
    else:
        correction_df.to_csv("correction_dataframe_custom.csv") 

        
#NOT NEEDED
def get_raw_data_results(df):
    
    ## converting output to numeric values
    le.fit(df["Pose"])
    df["label"] = le.transform(df["Pose"])
    
    ## dropping Pose column
    df.drop(columns=["Pose","ImgNum"],inplace=True)
    
    ## feature designing for train and test
    df #= get_designed_data_df(df)
    
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.25, random_state=0)

    """dataset = pd.concat([x_train, y_train], axis=1)
    dict1 = {} #pose -> 132 size array with mean keypoints

    for pose in range(0,81):
      temp = dataset.loc[dataset['label'] == pose].mean()
      dict1[pose] = temp
    print(dict1)"""

    
    ## running different models on this
    model_dict = run_models(x_train, x_test, y_train, y_test,False)
    
    return model_dict
    ## get cosine similarity results
    #get_cosine_similarity_results(x_train, x_test, y_train, y_test, df.columns)
    
def get_angle(p1, p2, p3):
    
    """
        returns angle between p1,p2,p3 with p2 as the pivot
    """
    
    v1 = p1-p2
    v2 = p3-p2
    
    num = np.inner(v1,v2)
    den = (np.sqrt((v1**2).sum())) * (np.sqrt((v2**2).sum()))
    
    return np.degrees(np.arccos(num/den))

def get_designed_data_df(df):
    
    d = {}

    for i in tqdm(range(df.shape[0])):

        j=0
        while j<df.shape[1]-1:

            col_name = df.columns[j].split("-")[0]

            data_point = np.array([df.iloc[i,j], df.iloc[i,j+1], df.iloc[i,j+2]])
            col_name = col_name.lower()

            if col_name not in d:
                d[col_name] = []

            d[col_name].append(data_point)
            j+=4

        if "label" not in d:
            d["label"] = []
        d["label"].append(df.iloc[i,j]) 
        
    return get_custom_features(pd.DataFrame(d))

def get_plane(p1, p2, p3):
    #equation of plane : ax + by + cz + d = 0
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p1[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * p1[0] - b * p1[1] - c * p1[2])

    return [a,b,c,d]

def get_custom_features(df):
    
    ## add features

    d = {}

    ## angles
    d["max_hand_angle"] = []
    d["min_hand_angle"] = []
    d["max_foot_angle"] = []
    d["min_foot_angle"] = []

    d["max_elbow_to_knee_angle"] = []
    d["min_elbow_to_knee_angle"] = []

    d["knee_to_knee_angle"] = []
    d["elbow_to_elbow_angle"] = []

    #d["heel_at_nose_angle"] = []
    #d["hand_at_nose_angle"] = []
    d["max_nose_to_heel_angle"] = []

    ## distances
    d["feet_to_shoulder_ratio"] = []
    d["hand_to_shoulder_ratio"] = []

    # symmetry
    """d["wrist_sagittal"] = []
    d["elbow_sagittal"] = []
    d["knee_sagittal"] = []
    d["ankle_sagittal"] = []"""

    for i in tqdm(range(df.shape[0])):

        # wrist symmetry : ratio of distances of left_wrist and right_wrist from sagittal plane
        """sagittal_plane = get_plane(df["nose"][i], (df["left_shoulder"][i]+df["right_shoulder"][i])/2, (df["left_hip"][i]+df["right_hip"][i])/2)
        denom = pow(pow(sagittal_plane[0],2) + pow(sagittal_plane[1],2) + pow(sagittal_plane[2],2), 0.5)

        dist_wrist_left = abs(sagittal_plane[0]*df["left_wrist"][i][0] + sagittal_plane[1]*df["left_wrist"][i][1] + sagittal_plane[2]*df["left_wrist"][i][2] + sagittal_plane[3]) / denom
        dist_wrist_right = abs(sagittal_plane[0]*df["right_wrist"][i][0] + sagittal_plane[1]*df["right_wrist"][i][1] + sagittal_plane[2]*df["right_wrist"][i][2] + sagittal_plane[3]) / denom
        d["wrist_sagittal"] = dist_wrist_left / dist_wrist_right

        dist_elbow_left = abs(sagittal_plane[0]*df["left_elbow"][i][0] + sagittal_plane[1]*df["left_elbow"][i][1] + sagittal_plane[2]*df["left_elbow"][i][2] + sagittal_plane[3]) / denom
        dist_elbow_right = abs(sagittal_plane[0]*df["right_elbow"][i][0] + sagittal_plane[1]*df["right_elbow"][i][1] + sagittal_plane[2]*df["right_elbow"][i][2] + sagittal_plane[3]) / denom
        d["elbow_sagittal"] = dist_elbow_left / dist_elbow_right

        dist_knee_left = abs(sagittal_plane[0]*df["left_knee"][i][0] + sagittal_plane[1]*df["left_knee"][i][1] + sagittal_plane[2]*df["left_knee"][i][2] + sagittal_plane[3]) / denom
        dist_knee_right = abs(sagittal_plane[0]*df["right_knee"][i][0] + sagittal_plane[1]*df["right_knee"][i][1] + sagittal_plane[2]*df["right_knee"][i][2] + sagittal_plane[3]) / denom
        d["knee_sagittal"] = dist_knee_left / dist_knee_right

        dist_ankle_left = abs(sagittal_plane[0]*df["left_ankle"][i][0] + sagittal_plane[1]*df["left_ankle"][i][1] + sagittal_plane[2]*df["left_ankle"][i][2] + sagittal_plane[3]) / denom
        dist_ankle_right = abs(sagittal_plane[0]*df["right_ankle"][i][0] + sagittal_plane[1]*df["right_ankle"][i][1] + sagittal_plane[2]*df["right_ankle"][i][2] + sagittal_plane[3]) / denom
        d["ankle_sagittal"] = dist_ankle_left / dist_ankle_right"""

        ## hand angles
        angle1 = get_angle(df["left_wrist"][i], df["left_elbow"][i], df["left_shoulder"][i])
        angle2 = get_angle(df["right_wrist"][i], df["right_elbow"][i], df["right_shoulder"][i])

        d["max_hand_angle"].append(max(angle1,angle2))
        d["min_hand_angle"].append(min(angle1,angle2))

        ## leg angles
        angle1 = get_angle(df["left_ankle"][i], df["left_knee"][i], df["left_hip"][i])
        angle2 = get_angle(df["right_ankle"][i], df["right_knee"][i], df["right_hip"][i])

        d["max_foot_angle"].append(max(angle1,angle2))
        d["min_foot_angle"].append(min(angle1,angle2))

        ## hand to leg angles
        # wrist to leg angle instead of elbow to knee
        angle1 = get_angle(df["left_elbow"][i], df["left_hip"][i], df["left_knee"][i])
        angle2 = get_angle(df["right_elbow"][i], df["right_hip"][i], df["right_knee"][i])
        #angle1 = get_angle(df["left_wrist"][i], df["left_hip"][i], df["left_ankle"][i])
        #angle2 = get_angle(df["right_wrist"][i], df["right_hip"][i], df["right_ankle"][i])

        d["max_elbow_to_knee_angle"].append(max(angle1,angle2))
        d["min_elbow_to_knee_angle"].append(min(angle1,angle2))

        ## elbow to elbow
        # try elbow-centroid-elbow instead
        mid_point = (df["left_shoulder"][i] + df["right_shoulder"][i]) / 2
        angle = get_angle(df["left_elbow"][i], mid_point, df["right_elbow"][i])

        #centroid = np.mean(df.iloc[i])
        #angle = get_angle(df["left_elbow"][i], centroid, df["right_elbow"][i])

        d["elbow_to_elbow_angle"].append(angle)

        ## knee to knee
        mid_point = (df["left_hip"][i] + df["right_hip"][i]) / 2
        angle = get_angle(df["left_knee"][i], mid_point, df["right_knee"][i])

        #angle = get_angle(df["left_knee"][i], centroid, df["right_knee"][i])
        d["knee_to_knee_angle"].append(angle)

        ## heels at nose and hand at nose angle
        # heel-waist-heel instead of heel-nose-heel
        #angle1 = get_angle(df["left_wrist"][i], df["nose"][i], df["right_wrist"][i])
        #angle2 = get_angle(df["left_heel"][i], df["nose"][i], df["right_heel"][i])
        #mid_point = (df["left_hip"][i] + df["right_hip"][i]) / 2
        #angle2 = get_angle(df["left_heel"][i], mid_point, df["right_heel"][i])

        #d["hand_at_nose_angle"].append(angle1)
        #d["heel_at_nose_angle"].append(angle2)

        # nose-waist-heel
        midpoint = (df["left_hip"][i] + df["right_hip"][i]) / 2
        angle1 = get_angle(df["nose"][i], midpoint, df["right_heel"][i])
        angle2 = get_angle(df["nose"][i], midpoint, df["left_heel"][i])

        d["max_nose_to_heel_angle"].append(max(angle1,angle2))

        ############################################
        
        ## centroid distance 
        centroid = np.mean(df.iloc[i])
        incl_cols = ["left_wrist", "left_elbow", "left_shoulder", "left_eye", "left_hip", "left_knee", "left_ankle",
                     "right_wrist", "right_elbow", "right_shoulder", "right_eye", "right_hip", "right_knee", "right_ankle"]
        for col in df.columns:
  
            if col in incl_cols:
                
                name = col+"_centroid_distance"
                if name not in d:
                    d[name] = []

                #print(col, df[col][i])
                d[name].append(np.sqrt(np.sum(np.square(df[col][i]-centroid))))
        
        ## feet and hand to shoulder ratio

        d_shoulder = np.sqrt(np.sum(np.square(df["left_shoulder"][i] - df["right_shoulder"][i])))
        d_hand = np.sqrt(np.sum(np.square(df["left_wrist"][i] - df["right_wrist"][i])))
        d_feet = np.sqrt(np.sum(np.square(df["left_foot_index"][i] - df["right_foot_index"][i])))

        d["feet_to_shoulder_ratio"].append(d_feet/d_shoulder)
        d["hand_to_shoulder_ratio"].append(d_hand/d_shoulder)

    ## output
    if "label" in df:
        d["label"] = df["label"]
    
    return pd.DataFrame(d)

# def get_designed_data_results(df):
    
#     ## converting output to numeric values
#     le.fit(df["Pose"])
#     df["label"] = le.transform(df["Pose"])
    
#     ## dropping Pose column
#     df.drop(columns=["Pose","ImgNum"],inplace=True)
    
   
#     ## feature designing for train and test
#     df = get_designed_data_df(df)
    
#     x = df.iloc[:,:-1]
#     y = df.iloc[:,-1]
    
#     x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.25, random_state=0)
    
    
    
#     model_dict = None
#     ## run models
#     model_dict = run_model(x_train, x_test, y_train, y_test,True)
    
#     ## cosine similarity
#     #get_cosine_similarity_results(x_train, x_test, y_train, y_test, df.columns)
    
#     return model_dict

def predict_image(img_path, model, le, design_features=True):
    
    ## get image keypoints
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        
        landmark_names = []
        
        for name in mp_pose.PoseLandmark:
            name = str(name).lower().split(".")[-1]
            landmark_names.append(name)
            
        img = cv2.imread(img_path)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      
        row = {}
        for i in range(0,33):
            coord = results.pose_landmarks.landmark[i]
  
            if landmark_names[i] not in row:
                row[landmark_names[i]] = []
            
            row[landmark_names[i]].append(np.array([coord.x, coord.y, coord.z]))
        
    df = pd.DataFrame(row)
    
    if design_features:
        
        df = get_custom_features(df)
        
    
    #val = model.predict_proba(df)[0][model.predict(df)]
    
    #if val < 5/82:
     #   print("PREDICTION UNCERTAIN")
        
    name = le.inverse_transform(model.predict(df))
    print("predicted_pose:",name)
    print("getting cosine similarity results...")
    y_test = ["" for i in range(df.shape[0])]
    y_pred = model.predict(df)

    cosine_eval(df, [1 for i in range(df.shape[0])], y_pred, df.columns)
