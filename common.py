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
    print("this the y type", type(y))
    y_train_trip_duration = y['trip_duration']
    return np.log1p(y_train_trip_duration).rename('log_'+y_train_trip_duration.name)

def preprocess_data(X):
    print(f"Preprocessing data")
    X['pickup_date'] = X['pickup_datetime'].dt.date
    dates = X['pickup_date'].sort_values()
    df_abnormal_dates = X.groupby('pickup_date').size()
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < df_abnormal_dates.quantile(0.02)]
    dict_weekday = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    weekday = X['pickup_datetime'].dt.weekday.map(dict_weekday).rename('weekday')
    hourofday = X['pickup_datetime'].dt.hour.rename('hour')
    month = X.pickup_datetime.dt.month.rename('month')
    res = X.copy()
    res['weekday'] = res['pickup_datetime'].dt.weekday
    res['month'] = res['pickup_datetime'].dt.month
    res['hour'] = res['pickup_datetime'].dt.hour
    res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
    return res

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
