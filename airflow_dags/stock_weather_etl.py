from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
import logging
import requests
import time
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
import boto3

# Default arguments untuk DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['athayaharmanaputri@mail.ugm.ac.id'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def fetch_weather_data(**kwargs):
    logging.info("Starting fetch_weather_data task...")
    
    # URL baru untuk API
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Bakersfield%2C%20California/2023-01-01/2024-11-19"
    params = {
        "unitGroup": "us",  # Gunakan unit 'us' sesuai dengan URL baru
        "key": "ZD25UARK6D5KZXZ7QL2AUD5DZ",  # API Key baru
        "contentType": "json"  # Format JSON
    }
    
    # Konfigurasi retry
    retries = 5
    for attempt in range(retries):
        try:
            logging.info(f"Attempt {attempt+1} to fetch data...")
            response = requests.get(url, params=params)
            
            # Log status code untuk debugging
            logging.info(f"Received status code: {response.status_code}")
            
            if response.status_code == 429:
                # Jika rate limit, gunakan exponential backoff
                wait_time = 2 ** attempt + 5
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            # Raise error untuk status code non-200
            response.raise_for_status()
            logging.info("Data fetched successfully.")
            
            # Parse data JSON
            data = response.json()["days"]
            df = pd.DataFrame(data)[["datetime", "temp", "dew", "humidity", "precip", "windspeed", "conditions"]]
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            # Simpan data ke file CSV sementara
            file_path = "/tmp/weather_data.csv"
            df.to_csv(file_path, index=False)
            logging.info(f"Weather data saved to {file_path}")
            
            # Return file path untuk XCom
            return file_path
        except Exception as e:
            logging.error(f"Error occurred during fetch: {e}")
            if attempt == retries - 1:
                raise Exception("Max retries exceeded") from e

# Fungsi untuk mengextract data saham
def fetch_stock_data(**kwargs):
    symbol = "TRC"
    api_key = "WN3N2YITLUYA6QXG"
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 11, 19)
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'datatype': 'json'
    }
    all_records = []
    current_date = end_date

    while current_date >= start_date:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            for date, values in time_series.items():
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                if start_date <= date_obj <= end_date:
                    all_records.append({
                        'date': date,
                        'open': values['1. open'],
                        'high': values['2. high'],
                        'low': values['3. low'],
                        'close': values['4. close'],
                        'volume': values['5. volume']
                    })
        else:
            break
    file_path = "/tmp/stock_data.csv"
    pd.DataFrame(all_records).to_csv(file_path, index=False)
    return file_path

# Fungsi untuk menyimpan data ke PostgreSQL
def load_to_postgres(**kwargs):
    ti = kwargs['task_instance']
    weather_file = ti.xcom_pull(task_ids='fetch_weather_data')
    stock_file = ti.xcom_pull(task_ids='fetch_stock_data')
    
    pg_hook = PostgresHook(postgres_conn_id="postgres_conn")
    
    weather_df = pd.read_csv(weather_file)
    stock_df = pd.read_csv(stock_file)
    
    pg_hook.copy_expert("COPY weather_data FROM stdin WITH CSV HEADER", weather_file)
    pg_hook.copy_expert("COPY stock_data FROM stdin WITH CSV HEADER", stock_file)

# Fungsi untuk menggabungkan data dan memuat ke S3
def join_and_upload_to_s3(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id="postgres_conn")
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    query = """
        SELECT
            w.datetime AS weather_date, w.temp, w.humidity, w.windspeed, w.conditions,
            s.date AS stock_date, s.open, s.high, s.low, s.close, s.volume
        FROM
            weather_data w
        JOIN
            stock_data s
        ON
            w.datetime = s.date
    """
    cursor.execute(query)
    joined_data = cursor.fetchall()
    
    joined_df = pd.DataFrame(joined_data, columns=[
        "weather_date", "temp", "humidity", "windspeed", "conditions",
        "stock_date", "open", "high", "low", "close", "volume"
    ])
    
    file_path = "/tmp/joined_data.csv"
    joined_df.to_csv(file_path, index=False)
    
    s3 = boto3.client('s3')
    s3.upload_file(file_path, "etl-awsbucket", "joined_data.csv")

# Definisi DAG
with DAG(
    'etl_weather_stock_pipeline',
    default_args=default_args,
    description='ETL pipeline for weather and stock data',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    
    start_pipeline = EmptyOperator(task_id='start_pipeline')
    
    fetch_weather = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data
    )
    
    fetch_stock = PythonOperator(
        task_id='fetch_stock_data',
        python_callable=fetch_stock_data
    )
    
    load_postgres = PythonOperator(
        task_id='load_to_postgres',
        python_callable=load_to_postgres
    )
    
    join_upload_s3 = PythonOperator(
        task_id='join_and_upload_to_s3',
        python_callable=join_and_upload_to_s3
    )
    
    end_pipeline = EmptyOperator(task_id='end_pipeline')

    # Task dependencies
    start_pipeline >> [fetch_weather, fetch_stock] >> load_postgres >> join_upload_s3 >> end_pipeline
