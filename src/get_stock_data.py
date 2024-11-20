# Alpha Ventaga API

import requests
import pandas as pd
from datetime import datetime, timedelta

# Simbol saham perusahaan pertanian: Tejon Ranch
symbol = 'TRC'  
api_key_stock = 'WN3N2YITLUYA6QXG'  #API Key Alpha Ventega API

# Fungsi untuk mengambil data saham harian dalam rentang tanggal yang diberikan
def get_stock_data(symbol, api_key, start_date, end_date):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full',  # Mengambil seluruh data yang tersedia
        'datatype': 'json'
    }

    all_records = []
    current_date = end_date

    while current_date >= start_date:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                records = []
                for date, values in time_series.items():
                    date_obj = datetime.strptime(date, '%Y-%m-%d')
                    if start_date <= date_obj <= end_date:
                        record = {
                            'date': date,
                            'open': values['1. open'],
                            'high': values['2. high'],
                            'low': values['3. low'],
                            'close': values['4. close'],
                            'volume': values['5. volume']
                        }
                        records.append(record)
                
                all_records.extend(records)

                # Update the current_date to continue fetching older data if needed
                current_date = datetime.strptime(list(time_series.keys())[-1], '%Y-%m-%d') - timedelta(days=1)
            else:
                print(f"Data tidak ditemukan untuk {symbol}")
                break
        else:
            print(f"Error: {response.status_code}")
            break

    # Reverse the records to make the earliest date first
    return all_records[::-1]

# Tentukan rentang tanggal yang diinginkan
start_date = datetime(2023, 1, 1)  # Mulai dari 1 Januari 2023
end_date = datetime(2024, 11, 19)  # Sampai dengan 19 November 2024

# Ambil data saham dalam rentang tanggal tersebut
stock_records = get_stock_data(symbol, api_key_stock, start_date, end_date)

# Simpan data saham dalam DataFrame
df_stock = pd.DataFrame(stock_records)

# Tampilkan data teratas
print(df_stock.head())

# Menyimpan data ke file CSV jika diinginkan
df_stock.to_csv('tejon_ranch_stock_data_2023_2024.csv', index=False)
