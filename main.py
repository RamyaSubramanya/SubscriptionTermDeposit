from src.pipeline import load_and_prepare_data, split_train_test
from src.modelling import plot_roc_auc_curve, train_and_evaluate_model
import xgboost
from xgboost import XGBClassifier

#Step1: load and prepare data
data = load_and_prepare_data(file_path=r"D:\Data Science\Machine Learning & Deep Learning ANN (Regression & Classification)\Classification Practicals\Subscription_to_TermDeposit\VSCode\data\bank-full.csv",
                           target_column='y')
print("Data has been loaded")

#Step2: split the data
X_train, X_test, y_train, y_test = split_train_test(data=data, target_column='y')
print("Data has been split into train and test")

#step3: build model 
xgb_predictions, rf_predictions, xgb_accuracy, xgb_recall, rf_accuracy, rf_recall = train_and_evaluate_model(X_train, X_test, y_train, y_test)
