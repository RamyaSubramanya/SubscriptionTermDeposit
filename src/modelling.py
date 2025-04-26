import matplotlib.pyplot as mp
import seaborn as sb

import sklearn
import xgboost
import mlflow

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier


def plot_roc_auc_curve(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:,1]

        
    #extract probabilities of class 1 only - focus is on positive classes
    y_probs = model.predict_proba(X_test)[:,1]    

    #roc_curve results in fpr, tpr values and threshold, let's save fpr, tpr only
    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_probs)      

    #calculate auc 
    auc_value = round(auc(fpr,tpr),2)
    # print(auc_value)

    #plot roc, auc curve
    #define the size of the plot
    mp.figure(figsize=(5,3))

    #feed in values - fpr, tpr, auc as text format
    mp.plot(fpr, tpr, color='blue', label='ROC Curve')
    mp.plot([0,1], [0,1], color='gray', linestyle='--', label='AUC')
    mp.text(x=0.38, y=0.6, s=f'AUC={auc_value}', fontsize=10)

    #label the plot on x-axis and y-axis
    mp.title("ROC-AUC curve", fontsize=11)
    mp.xlabel('False Positive Rate (FPR)')
    mp.ylabel('True Positive Rate (TPR)')
    mp.legend(loc='lower right')
    mp.show()



def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    #log mlflow parameters
    mlflow.start_run()
        
    #Initiatlize 2 models - Gradient Boosting and Random Forest model 
    xgb_model = XGBClassifier(n_estimators=150)
    rf_model = RandomForestClassifier(n_estimators=250)

    #train the models 
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    #make predictions on test data 
    xgb_predictions = xgb_model.predict(X_test)
    rf_predictions = rf_model.predict(X_test)

    #Performance metrics
    xgb_accuracy = round(accuracy_score(y_test, xgb_predictions),2)
    xgb_recall = round(recall_score(y_test, xgb_predictions),2)

    rf_accuracy = round(accuracy_score(y_test, rf_predictions),2)
    rf_recall = round(recall_score(y_test, rf_predictions),2)

    #log hyperparameters
    mlflow.log_param("xgb_n_estimators", 150)
    mlflow.log_param("rf_n_estimators", 250)

    #log error metric
    mlflow.log_metric("xgb_accuracy", xgb_accuracy)
    mlflow.log_metric("xgb_recall", xgb_recall)
    mlflow.log_metric("rf_accuracy", rf_accuracy)
    mlflow.log_metric("rf_recall", rf_recall)

    # End the MLflow run
    mlflow.end_run()
    
    #ROC-AUC curve
    plot_roc_auc_curve(xgb_model)    
    plot_roc_auc_curve(rf_model)    
    return xgb_predictions, rf_predictions, xgb_accuracy, xgb_recall, rf_accuracy, rf_recall

