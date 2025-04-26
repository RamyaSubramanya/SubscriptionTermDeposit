from src.pipeline import load_and_prepare_data, split_train_test
from src.modelling import plot_roc_auc_curve, train_and_evaluate_model
import pandas as pd

def test_model():
    data = load_and_prepare_data(file_path="tests\bank-full.csv",
                                    target_column='y')
    X_train, X_test, y_train, y_test = split_train_test(data=data, target_column='y')
    xgb_predictions, rf_predictions, xgb_accuracy, xgb_recall, rf_accuracy, rf_recall = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    assert len(rf_predictions) == len(y_test)
    assert len(xgb_predictions) == len(y_test)
    assert isinstance(xgb_accuracy, float)
    assert isinstance(rf_accuracy, float)
    print("Testing has been completed.")