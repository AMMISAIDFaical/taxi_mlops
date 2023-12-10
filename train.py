import sqlite3

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import common

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    X_train = pd.read_sql('SELECT * FROM X_train', con, parse_dates=['pickup_datetime'])
    y_train = pd.read_sql('SELECT * FROM y_train', con)
    con.close()
    # print(X_train['pickup_datetime'].dtype) #the pickup_datetime string convert issue (solved)
    print(type(X_train.dtypes))
    type(y_train)
    return X_train, y_train

def model_train_pipline(num_features,cat_features,X_train, y_train):
    train_features = num_features + cat_features
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)]
    )
    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])
    model = pipeline.fit(X_train[train_features], y_train)
    y_pred_train = model.predict(X_train[train_features])
    print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))
    score = mean_squared_error(y_train, y_pred_train)
    print(f"Score on train data {score:.2f}")
    return model

def model_ft_engen_1(X_train, y_train):
    num_features = ['abnormal_period', 'hour']
    cat_features = ['weekday', 'month']
    print("model ft engineered 1")
    model = model_train_pipline(num_features,cat_features,X_train, y_train)
    return model

def model_ft_engen_2(X_train, y_train):
    num_features = ['log_distance_haversine', 'hour',
                    'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                    'is_rare_pickup_point', 'is_rare_dropoff_point']
    cat_features = ['weekday', 'month']
    print("model ft engineered 2")
    model = model_train_pipline(num_features, cat_features,X_train, y_train)
    return model

if __name__ == "__main__":
    X_train, y_train = load_train_data(common.DB_PATH)
    abnormal_dates,X_train = common.preprocess_data(X_train)
    y_train = common.transform_target(y_train)
    X_train = common.ftEngen_step_1(X_train,abnormal_dates)
    model_1 = model_ft_engen_1(X_train, y_train)
    X_train = common.ftEngen_step_2(X_train)
    model_2 = model_ft_engen_2(X_train, y_train)
    common.persist_model(model_1, common.MODEL_PATH,"model_1")
    # common.persist_model(model_2, common.MODEL_PATH,'model_2')