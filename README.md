# Stock-Tejon-Ranch-DataEngineer
End to End Data Engineering Project (B)

This project fetches stock data from Alpha Vantage and weather data from Visual Crossing for Tejon Ranch located in Bakersfield, CA. The data is then processed and stored in AWS S3.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/username/tejon-ranch-project.git
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
4. Set up Airflow to run he ETL DAG on a schedule:
   ```bash
   airflow dags trigger stock_weather_etl

## API Keys
1. Alpha antage API Key: [link](https://www.alphavantage.co/)
2. Visual Crossing API Key: [link](https://www.visualcrossing.com/)

## Member Group
- Athaya Harmana Putri -(NIM)
- Danella Zefanya Siahaan -(NIM)
- Satama Safika -22/492880/TK/53955
