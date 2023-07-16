#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import pandas as pd
import boto3


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


def save_data(df, output_file, options):
    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def prepare_data(df, categorical):

    df = df.copy()

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# Make categorical a parameter for read_data and pass it inside main
def read_data(filename, categorical):


    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', "http://localhost:4566/")
    s3_client = boto3.client('s3', endpoint_url = s3_endpoint_url)

    if s3_endpoint_url:

        options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }}

        df = pd.read_parquet('s3://nyc-duration/in/2022-01.parquet', storage_options=options)

  
    # Si la URL del endpoint de LocalStack no está establecida, lee el archivo de la manera usual
    else:
        
        df = pd.read_parquet(filename)

    
    df = prepare_data(df, categorical)
    
    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

# Let's create a function main with two parameters: year and month.
def main(year, month):

    # Move all the code (except read_data) inside main
    categorical = ['PULocationID', 'DOLocationID']
    
    #input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

    input_file = get_input_path(year, month)
    print(input_file)
    output_file = get_output_path(year, month)
    print(output_file)

    # Make categorical a parameter for read_data and pass it inside main
    
    df = read_data(input_file, categorical)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    # Calcula la suma de las duraciones predichas para el DataFrame de prueba
    sum_of_predicted_durations = df_result['predicted_duration'].sum()
    print(f"The sum of predicted durations for the test dataframe is {sum_of_predicted_durations}")


    #df_result.to_parquet(output_file, engine='pyarrow', index=False)
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', "http://localhost:4566/")
    s3_client = boto3.client('s3', endpoint_url = s3_endpoint_url)
    options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
    save_data(df_result, "s3://nyc-duration/out/2022-01.parquet", options)


if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)   
