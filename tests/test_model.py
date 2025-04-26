from src.pipeline import load_and_prepare_data, split_train_test
from src.modelling import plot_roc_auc_curve, train_and_evaluate_model
import pandas as pd
import os

def test_model():
    current_dir = os.getcwd()
    csv_path = os.path.join(current_dir, 'tests', 'bank-full.csv')
    data = load_and_prepare_data(file_path=csv_path, target_column='y')
    X_train, X_test, y_train, y_test = split_train_test(data=data, target_column='y')
    xgb_predictions, rf_predictions, xgb_accuracy, xgb_recall, rf_accuracy, rf_recall = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    assert len(rf_predictions) == len(y_test)
    assert len(xgb_predictions) == len(y_test)
    assert isinstance(xgb_accuracy, float)
    assert isinstance(rf_accuracy, float)
    print("Testing has been completed.")