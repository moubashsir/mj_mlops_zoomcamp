#!/usr/bin/env python
# coding: utf-8



#get_ipython().system('pip freeze | grep scikit-learn')

import pickle
import pandas as pd
import pyarrow
import numpy
import os
import sys

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    df = pd.read_parquet(filename)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

#dv, model = load_model()
#dicts = df[categorical].to_dict(orient='records')
#X_val = dv.transform(dicts)
#y_pred = model.predict(X_val)


#numpy.std(y_pred)

def apply_model():
    
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    print("Getting the path of the files")
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'yellow_taxi_scored_{year:04d}-{month:02d}.parquet'

    print("Reading data")
    df = read_data(input_file)

    print("Loading Model")
    dv, model = load_model()
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    print("Starting prediction")
    y_pred = model.predict(X_val)
    print("Prediction complete")
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    print("Saving output file")
    print(numpy.mean(y_pred))
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


if __name__ == '__main__':
    apply_model()

