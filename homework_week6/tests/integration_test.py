import os
import boto3

import pandas as pd
from datetime import datetime


s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', "http://localhost:4566/")
s3_client = boto3.client('s3', endpoint_url = s3_endpoint_url)

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_integration_test():
    
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]
    categorical = ['PULocationID', 'DOLocationID']
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)
    options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}

    input_file = "s3://nyc-duration/in/2022-01.parquet"
    
    # Guarda el DataFrame en S3
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    # Ejecuta el script batch.py
    os.system("python batch.py 2022 1")


    # Lee el archivo de S3
    df_output = pd.read_parquet(input_file, storage_options=options)
   

    # Comprueba que el DataFrame de salida es igual al DataFrame de entrada
    assert pd.testing.assert_frame_equal(df_input, df_output) == None

