import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

def load_and_prepare_data(file_path, target_column):
    """
    Load data from the file_path
    Convert target column to integer encoding (subscription = 1, non-subscription=0)
    Convert categorical into dummies
    """
    data = pd.read_csv(file_path)

    #target variable set to 1 or 0 instead of yes and no
    data[target_column] = data[target_column].apply(lambda x: 1 if x=='yes' else 0)
    
    #convert categorical into dummies
    data = pd.get_dummies(data, drop_first=True, dtype='int')
    return data

def split_train_test(data, target_column):
    """
    Split the data into training and test data with a split ratio of 70:30
    """
    if target_column not in data.columns:
        raise ValueError(f'{target_column} is not found in the dataset.')

    X = data.drop(columns=[target_column])
    y = data[target_column]

    #train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test