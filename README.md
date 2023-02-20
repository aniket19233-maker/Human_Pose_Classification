# Human_Pose_Classification

To run the repository use colab and clone the repository and after cloning run the following command
```
%cd Human_Pose_Classification
! gdown --id 10kbGdYv0YylbEup_z__clDDLpSv9FUY2
! gdown --id 1--y6f1d3I1ai6FKIHQmYO-ECgBKp28NX
! gdown -- id 1bv_M9vVC-mn9qmNGZVIsuI7inO-SUHju
```

To install the dependencies pls run the following command
```
!python3 -m pip install -r config.txt
```
For training the model run the following command
```
!python3 train.py --model= <MODEL_NAME> --feature_type = <FEATURE_TYPE>
```

'MODEL_NAME' values

LR = Logistic Regression Classifier<br />
GNB = Gaussian Naive Bayes<br />
RF = Random Forest Classifier<br />
XGB = XGBoost Classifier<br />
ADA = Adaboost Classifier<br />
SGD = SGD Classifier<br />
SVM = Support Vector Machine Classifier

'FEATURE_TYPE' values

raw = to feed raw keypoints data to the models <br/>
custom = to feed custom features designed from raw keypoints to the models
