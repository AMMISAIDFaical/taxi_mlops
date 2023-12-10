import sqlite3
from datetime import datetime
from typing import Union
from typing_extensions import Annotated
import numpy as np
import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, PositiveInt, Field
import common

app = FastAPI()
# {
#   "vendor_id": 1,
#   "pickup_datetime": "2023-12-08T12:00:00",
#   "passenger_count": 2,
#   "pickup_longitude": -73.9876,
#   "pickup_latitude": 40.7488,
#   "dropoff_longitude": -74.0060,
#   "dropoff_latitude": 40.7128,
#   "store_and_fwd_flag": "Y"
# }

class Trip(BaseModel):
    vendor_id: PositiveInt = Annotated[int, Field(gt=0)]
    pickup_datetime: datetime
    passenger_count: PositiveInt = Annotated[int, Field(gt=0,lt=5)]
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float

# Assume you have a model loaded, e.g., `taxi_model`
taxi_model = common.load_model(common.MODEL_PATH)  # Load or instantiate your machine learning model

def pred_req_DbSave(trip_prediction_df, pred_req_history):
    print(f"Saving Prediction and Request in Db: {common.DB_PATH}")

    # Creating a connection to the SQLite database
    with sqlite3.connect(common.DB_PATH) as con:
        cur = con.cursor()
        # Creating column list for insertion
        # creating column list for insertion
        cols = ",".join([str(i) for i in trip_prediction_df.columns.tolist()])
        for i, row in trip_prediction_df.iterrows():
            # Convert datetime object to string format
            row['pickup_datetime'] = row['pickup_datetime'].strftime('%Y-%m-%d %H:%M:%S')
            # Building the SQL query with placeholders
            sql = f"INSERT INTO pred_req_history ({cols}) VALUES ({', '.join(['?' for _ in range(len(row))])})"
            print(sql)
            # Executing the SQL query with the values from the current row
            print(row)
            cur.execute(sql, tuple(row))
        # Committing changes to the database
        con.commit()

# Example usage:
# pred_req_DbSave(your_trip_prediction_df, "your_table_name")

def create_req_pred(req_pred_df):
    print(f"create Prediction and Request table in Db : {common.DB_PATH}")
    with sqlite3.connect(common.DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS pred_req_history")
        req_pred_df.to_sql(name='pred_req_history', con=con, if_exists="replace")
    return req_pred_df

@app.post("/predict_taxi/")
async def predict_taxi(request: Trip):
    print(type(request))
    print(request)
    # Convert the Pydantic model to a dictionary
    trip_dict = request.dict()
    # Create a Pandas DataFrame from the dictionary
    trip_df = pd.DataFrame([trip_dict])
    print(type(trip_df))
    print(trip_df)
    abnormal_dates,X = common.preprocess_data(trip_df)
    trip_df = common.ftEngen_step_1(X,abnormal_dates)
    # # Make predictions
    prediction = taxi_model.predict(trip_df)
    prediction = np.expm1((prediction[0]))
    pred_series = pd.Series(prediction, name='trip_duration', dtype=np.float64)
    # Concatenate DataFrame and Series
    infer_res = pd.concat([trip_df, pred_series], axis=1)
    req_pred_df = create_req_pred(infer_res)
    pred_req_DbSave(infer_res, req_pred_df)
    return {"prediction": prediction}


if __name__ == '__main__':
    uvicorn.run("fast_api_taxi:app", host="localhost",
                port=8000, reload=True)
