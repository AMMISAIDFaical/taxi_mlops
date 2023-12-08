from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field, PositiveFloat, computed_field,PositiveInt,condate,confloat
import common

app = FastAPI()

class Trip(BaseModel):
    vendor_id: int
    pickup_datetime: datetime
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float

# Assume you have a model loaded, e.g., `taxi_model`
taxi_model = common.load_model(common.MODEL_PATH)  # Load or instantiate your machine learning model

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
    trip_df = common.preprocess_data(trip_df)
    # # Make predictions
    # print(taxi_model)
    print(trip_df.columns)
    print(type(trip_df))
    prediction = taxi_model.predict(trip_df)
    print(np.expm1((prediction[0])))
    return {"prediction": np.expm1((prediction[0]))}

if __name__ == '__main__':
    uvicorn.run("fast_api_taxi:app", host="localhost",
                port=8000, reload=True)