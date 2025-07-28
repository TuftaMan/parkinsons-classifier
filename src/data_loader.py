import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_load(path):
    df = pd.read_csv(path)
    df = df.dropna()
    y = df['status']
    X = df.drop(['name', 'status'], axis=1)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
