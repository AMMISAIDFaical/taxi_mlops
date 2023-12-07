import pickle
import os

import numpy as np

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')

# Using INI configuration file
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)
DB_PATH = str(config.get("PATHS", "DB_PATH"))
MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))
DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))

def transform_target(y):
  return np.log1p(y).rename('log_'+y.name)
def step1_add_features(X,abnormal_dates):
  res = X.copy()
  res['weekday'] = res['pickup_datetime'].dt.weekday
  res['month'] = res['pickup_datetime'].dt.month
  res['hour'] = res['pickup_datetime'].dt.hour
  res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
  return res

def preprocess_data(X):
    print(f"Preprocessing data")
    X['pickup_date'] = X['pickup_datetime'].dt.date
    dates = X['pickup_date'].sort_values()
    df_abnormal_dates = X.groupby('pickup_date').size()
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < df_abnormal_dates.quantile(0.02)]
    X = step1_add_features(X)
    X_train = step1_add_features(X_train)
    X_test = step1_add_features(X_test)

    return X
def New_features(X):
    def step1_add_features(X):
        res = X.copy()
        res['weekday'] = res['pickup_datetime'].dt.weekday
        res['month'] = res['pickup_datetime'].dt.month
        res['hour'] = res['pickup_datetime'].dt.hour
        res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
        return res

    X = step1_add_features(X)
    X_train = step1_add_features(X_train)
    X_test = step1_add_features(X_test)
def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model
