import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error

import common

def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    X_test = pd.read_sql('SELECT * FROM X_test', con, parse_dates=['pickup_datetime'])
    y_test = pd.read_sql('SELECT * FROM y_test', con)
    con.close()
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    print(f"Evaluating the model")
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    return score

if __name__ == "__main__":
    X_test, y_test = load_test_data(common.DB_PATH)
    X_test = common.preprocess_data(X_test)
    y_test = common.transform_target(y_test)
    model = common.load_model(common.MODEL_PATH)
    score_test = evaluate_model(model, X_test, y_test)
    print(f"Score on test data {score_test:.2f}")