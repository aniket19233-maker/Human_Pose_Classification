# Human_Pose_Classification

To run the repository use colab and clone the repository and after cloning run the following command
```
%cd Human_Pose_Classification
! gdown --id 18WNBVg98h_JKXCSKaZDELalBXZkvExI3
```




To install the dependencies pls run the following command
```
!python3 -m pip install -r config.txt
```
For training the model run the following command
```
!python3 train.py --model= <MODEL_NAME>
```
Follow the 'MODEL_NAME' abbreviations for running the above command

LR = Logistic Regression Classifier
GNB = Gaussian Naive Bayes
RF = Random Forest Classifier
XGB = XGBoost Classifier
ADA = Adaboost Classifier
SGD = SGD Classifier
SVM = Support Vector Machine Classifier
