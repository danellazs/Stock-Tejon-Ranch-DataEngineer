import pandas as pd
import matplotlib.pyplot as plt

# Mengubah kolom 'date' menjadi tipe datetime
df_stock['date'] = pd.to_datetime(df_stock['date'])

# Mengonversi kolom yang mengandung angka ke tipe numerik (jika diperlukan)
df_stock['open'] = pd.to_numeric(df_stock['open'], errors='coerce')
df_stock['high'] = pd.to_numeric(df_stock['high'], errors='coerce')
df_stock['low'] = pd.to_numeric(df_stock['low'], errors='coerce')
df_stock['close'] = pd.to_numeric(df_stock['close'], errors='coerce')
df_stock['volume'] = pd.to_numeric(df_stock['volume'], errors='coerce')

# Menghitung perubahan harga harian (% perubahan harga penutupan)
df_stock['price_change'] = df_stock['close'].pct_change() * 100  # Menghitung perubahan persentase

# Menampilkan data pertama
print(df_stock)

# Visualisasi Pergerakan Harga (Harga Penutupan)
plt.figure(figsize=(10,6))
plt.plot(df_stock['date'], df_stock['close'], label='Close Price', color='blue', marker='o')
plt.title('Pergerakan Harga Saham Tejon Ranch (Penutupan)', fontsize=14)
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (USD)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# Visualisasi Volume Perdagangan
plt.figure(figsize=(10,6))
plt.bar(df_stock['date'], df_stock['volume'], color='green')
plt.title('Volume Perdagangan Saham Tejon Ranch', fontsize=14)
plt.xlabel('Tanggal')
plt.ylabel('Volume Perdagangan')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Visualisasi Perubahan Harga Harian (%)
plt.figure(figsize=(10,6))
plt.plot(df_stock['date'], df_stock['price_change'], label='Daily Price Change (%)', color='red', marker='o')
plt.title('Perubahan Harga Saham Tejon Ranch (Persentase Harian)', fontsize=14)
plt.xlabel('Tanggal')
plt.ylabel('Perubahan Harga (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Menghitung Q1, Q2, Q3 untuk data harga penutupan (close)
Q1 = df_stock['close'].quantile(0.25)
Q2 = df_stock['close'].quantile(0.50)  # Median
Q3 = df_stock['close'].quantile(0.75)

Q1_volume = df_stock['volume'].quantile(0.25)
Q2_volume = df_stock['volume'].quantile(0.50)  # Median
Q3_volume = df_stock['volume'].quantile(0.75)

Q1_price_change = df_stock['price_change'].quantile(0.25)
Q2_price_change = df_stock['price_change'].quantile(0.50)  # Median
Q3_price_change = df_stock['price_change'].quantile(0.75)

# Menghitung IQR dan menetapkan batas atas dan batas bawah
IQR = Q3 - Q1
IQR_volume = Q3_volume - Q1_volume
IQR_price_change = Q3_price_change - Q1_price_change
lower_bound = Q1 - (IQR * 3 / 4)  # Rumus yang diberikan
upper_bound = Q3 + (IQR * 3 / 4)

lower_bound_volume = Q1_volume - (IQR_volume * 3 / 4)  # Rumus yang diberikan
upper_bound_volume = Q3_volume + (IQR_volume * 3 / 4)

lower_bound_price_change = Q1_price_change - (IQR_price_change * 3 / 4)  # Rumus yang diberikan
upper_bound_price_change = Q3_price_change + (IQR_price_change * 3 / 4)

# Menampilkan batas bawah dan atas
print(f"Lower Bound (Close): {lower_bound}, Upper Bound (Close): {upper_bound}")
print(f"Lower Bound (Volume): {lower_bound_volume}, Upper Bound (Volume): {upper_bound_volume}")
print(f"Lower Bound (Price Change): {lower_bound_price_change}, Upper Bound (Price Change): {upper_bound_price_change}")

# Menyaring data untuk menghapus outlier
df_stock_no_outliers = df_stock[(df_stock['close'] >= lower_bound) & (df_stock['close'] <= upper_bound)]
df_stock_no_outliers_volume = df_stock[(df_stock['volume'] >= lower_bound_volume) & (df_stock['volume'] <= upper_bound_volume)]
df_stock_no_outliers_price_change = df_stock[(df_stock['price_change'] >= lower_bound_price_change) & (df_stock['price_change'] <= upper_bound_price_change)]

# Visualisasi data sebelum dan sesudah menghilangkan outlier
plt.figure(figsize=(12, 6))

# Plot harga penutupan sebelum menghilangkan outlier
plt.subplot(1, 2, 1)
plt.plot(df_stock['date'], df_stock['close'], marker='o', linestyle='-', color='b', label='Close Price')
plt.title('Saham - Sebelum Menghilangkan Outlier')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')

# Plot harga penutupan setelah menghilangkan outlier
plt.subplot(1, 2, 2)
plt.plot(df_stock_no_outliers['date'], df_stock_no_outliers['close'], marker='o', linestyle='-', color='g', label='Close Price (Tanpa Outlier)')
plt.title('Saham - Setelah Menghilangkan Outlier')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')

plt.tight_layout()
plt.show()

df_clean_clo = df_stock_no_outliers

# Visualisasi Volume Perdagangan sebelum dan sesudah menghilangkan outlier
plt.figure(figsize=(12, 6))

# Plot volume perdagangan sebelum menghilangkan outlier
plt.subplot(1, 2, 1)
plt.bar(df_stock['date'], df_stock['volume'], color='green')
plt.title('Volume Perdagangan - Sebelum Menghilangkan Outlier')
plt.xlabel('Tanggal')
plt.ylabel('Volume Perdagangan')
plt.xticks(rotation=45)
plt.grid(True)

# Plot volume perdagangan setelah menghilangkan outlier
plt.subplot(1, 2, 2)
plt.bar(df_stock_no_outliers_volume['date'], df_stock_no_outliers_volume['volume'], color='orange')
plt.title('Volume Perdagangan - Setelah Menghilangkan Outlier')
plt.xlabel('Tanggal')
plt.ylabel('Volume Perdagangan')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

df_clean_vol = df_stock_no_outliers_volume

# Visualisasi Perubahan Harga Harian sebelum dan sesudah menghilangkan outlier
plt.figure(figsize=(12, 6))

# Plot perubahan harga harian sebelum menghilangkan outlier
plt.subplot(1, 2, 1)
plt.plot(df_stock['date'], df_stock['price_change'], label='Daily Price Change (%)', color='red', marker='o')
plt.title('Perubahan Harga Harian - Sebelum Menghilangkan Outlier')
plt.xlabel('Tanggal')
plt.ylabel('Perubahan Harga (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Plot perubahan harga harian setelah menghilangkan outlier
plt.subplot(1, 2, 2)
plt.plot(df_stock_no_outliers_price_change['date'], df_stock_no_outliers_price_change['price_change'], label='Daily Price Change (%)', color='blue', marker='o')
plt.title('Perubahan Harga Harian - Setelah Menghilangkan Outlier')
plt.xlabel('Tanggal')
plt.ylabel('Perubahan Harga (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

df_clean_pri = df_stock_no_outliers_price_change

import pandas as pd

# Load the CSV file into a DataFrame
file_path = "/content/weather_bakersfield_2023_2024.csv"
df_weather = pd.read_csv(file_path)

# Display the first few rows of the dataframe to verify the data
df_weather.head()

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# 1. Label Encoding for conditions
label_encoder = LabelEncoder()
df_weather['conditions_encoded'] = label_encoder.fit_transform(df_weather['conditions'])

# 2. Apply PCA (Principal Component Analysis) to the features
features = df_weather[['temp', 'dew', 'humidity', 'precip', 'windspeed']]
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
pca_components = pca.fit_transform(features)

# 3. Add the reduced components and the encoded label to the DataFrame
df_weather['PCA1'] = pca_components[:, 0]
df_weather['PCA2'] = pca_components[:, 1]

# Display the DataFrame with reduced dimensions and encoded labels
print(df_weather[['datetime', 'conditions', 'conditions_encoded', 'PCA1', 'PCA2']])

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Label Encoding for conditions
label_encoder = LabelEncoder()
df_weather['conditions_encoded'] = label_encoder.fit_transform(df_weather['conditions'])

# 2. Apply PCA (Principal Component Analysis) to the features
features = df_weather[['temp', 'dew', 'humidity', 'precip', 'windspeed']]
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
pca_components = pca.fit_transform(features)

# 3. Add the reduced components and the encoded label to the DataFrame
df_weather['PCA1'] = pca_components[:, 0]
df_weather['PCA2'] = pca_components[:, 1]

# 4. Perform KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Change the number of clusters as needed
df_weather['cluster'] = kmeans.fit_predict(df_weather[['PCA1', 'PCA2']])

# 5. Visualize the Clustering Results
plt.figure(figsize=(10, 6))
plt.scatter(df_weather['PCA1'], df_weather['PCA2'], c=df_weather['cluster'], cmap='viridis', s=50)
plt.title('PCA Clustering of Weather Conditions')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(label='Cluster')
plt.show()

# Display the DataFrame with the cluster labels
print(df_weather[['datetime', 'conditions', 'conditions_encoded', 'PCA1', 'PCA2', 'cluster']])

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming df_weather and df_clean_vol are already loaded into dataframes

# Convert the 'datetime' column in weather data and 'date' column in stock data to datetime
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
df_clean_vol['date'] = pd.to_datetime(df_clean_vol['date'])

# Split the weather 'conditions' into individual conditions and perform one-hot encoding
weather_conditions = df_weather['conditions'].str.split(', ', expand=True).stack().unique()
for condition in weather_conditions:
    df_weather[condition] = df_weather['conditions'].str.contains(condition).astype(int)

# Merge the weather and stock data on date
merged_data_vol = pd.merge(df_weather, df_clean_vol, left_on='datetime', right_on='date', how='inner')

# Select relevant columns for correlation (weather-related and stock-related)
merged_data_vol = merged_data_vol[['datetime', 'temp', 'dew', 'humidity', 'precip', 'windspeed'] + list(weather_conditions) + ['open', 'high', 'low', 'close', 'volume']]

# Clustering: KMeans (as an example) to generate 'cluster' column
kmeans = KMeans(n_clusters=3, random_state=42)  # Set n_clusters to desired number of clusters
merged_data_vol['cluster'] = kmeans.fit_predict(merged_data_vol[['temp', 'dew', 'humidity', 'precip', 'windspeed']])

# Now we can generate correlation heatmaps for each cluster
for cluster_id in merged_data_vol['cluster'].unique():
    cluster_data = merged_data_vol[merged_data_vol['cluster'] == cluster_id]

    # Calculate correlation for each cluster
    correlation_matrix_cluster = cluster_data.corr()
    mask = np.triu(np.ones_like(correlation_matrix_cluster, dtype=bool))

    # Visualize heatmap for each cluster
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix_cluster, annot=True, mask=mask, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Heatmap for Cluster {cluster_id}')
    plt.show()

# Optional: You can also visualize the overall correlation matrix if needed
correlation_matrix = merged_data_vol.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create the heatmap with the lower triangle only for the overall dataset
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Lower Triangle Correlation Heatmap between Weather and Stock Data (Overall)')
plt.show()

# Pairplot to observe pairwise relationships
sns.pairplot(merged_data_vol[['temp', 'dew', 'humidity', 'precip', 'windspeed', 'open', 'close', 'volume']])
plt.suptitle('Pairplot: Pairwise Relationships Between Weather and Stock Data', y=1.02)
plt.show()

# Scatter plot with regression line for temperature vs stock closing price
sns.lmplot(x='temp', y='close', data=merged_data_vol, aspect=1.5, height=6, line_kws={'color': 'red'})
plt.title('Temperature vs Stock Close Price (with Regression Line)')
plt.show()

df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
df_clean_clo ['date'] = pd.to_datetime(df_clean_clo['date'])

# Split the weather 'conditions' into individual conditions and perform one-hot encoding
weather_conditions = df_weather['conditions'].str.split(', ', expand=True).stack().unique()
for condition in weather_conditions:
    df_weather[condition] = df_weather['conditions'].str.contains(condition).astype(int)

# Merge the weather and stock data on date
merged_data_clo = pd.merge(df_weather, df_clean_clo, left_on='datetime', right_on='date', how='inner')

# Select relevant columns for correlation (weather-related and stock-related)
merged_data_clo = merged_data_clo[['datetime', 'temp', 'dew', 'humidity', 'precip', 'windspeed'] + list(weather_conditions) + ['open', 'high', 'low', 'close', 'volume']]

# Calculate the correlation matrix
correlation_matrix = merged_data_clo.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create the heatmap with the lower triangle only
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Lower Triangle Correlation Heatmap between Weather and Stock Data')
plt.show()

# Pairplot to observe pairwise relationships
sns.pairplot(merged_data_clo[['temp', 'dew', 'humidity', 'precip', 'windspeed', 'open', 'close', 'volume']])
plt.suptitle('Pairplot: Pairwise Relationships Between Weather and Stock Data', y=1.02)
plt.show()

# Scatter plot with regression line for temperature vs stock closing price
sns.lmplot(x='temp', y='close', data=merged_data_clo, aspect=1.5, height=6, line_kws={'color': 'red'})
plt.title('Temperature vs Stock Close Price (with Regression Line)')
plt.show()

df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
df_clean_pri ['date'] = pd.to_datetime(df_clean_pri['date'])

# Split the weather 'conditions' into individual conditions and perform one-hot encoding
weather_conditions = df_weather['conditions'].str.split(', ', expand=True).stack().unique()
for condition in weather_conditions:
    df_weather[condition] = df_weather['conditions'].str.contains(condition).astype(int)

# Merge the weather and stock data on date
merged_data_pri = pd.merge(df_weather, df_clean_pri, left_on='datetime', right_on='date', how='inner')

# Select relevant columns for correlation (weather-related and stock-related)
merged_data_pri = merged_data_clo[['datetime', 'temp', 'dew', 'humidity', 'precip', 'windspeed'] + list(weather_conditions) + ['open', 'high', 'low', 'close', 'volume']]

# Calculate the correlation matrix
correlation_matrix = merged_data_clo.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create the heatmap with the lower triangle only
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Lower Triangle Correlation Heatmap between Weather and Stock Data')
plt.show()

# Pairplot to observe pairwise relationships
sns.pairplot(merged_data_pri[['temp', 'dew', 'humidity', 'precip', 'windspeed', 'open', 'close', 'volume']])
plt.suptitle('Pairplot: Pairwise Relationships Between Weather and Stock Data', y=1.02)
plt.show()

# Scatter plot with regression line for temperature vs stock closing price
sns.lmplot(x='temp', y='close', data=merged_data_pri, aspect=1.5, height=6, line_kws={'color': 'red'})
plt.title('Temperature vs Stock Close Price (with Regression Line)')
plt.show()

print(df_weather['datetime'].dtypes)  # Pastikan tipe ini adalah datetime64
print(df_clean_vol['date'].dtypes)    # Pastikan tipe ini juga datetime64

df_clean_vol = df_clean_vol.copy()
df_clean_vol['date'] = pd.to_datetime(df_clean_vol['date'], format='%Y-%m-%d')

print(df_clean_vol.head())
print(df_clean_vol.columns)

print(df_weather.head())
print(df_weather.columns)

df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], format='%Y-%m-%d')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Contoh: Data cuaca dan saham
# Data cuaca memiliki kolom: date, temp, dew, rain
# Data saham memiliki kolom: date, close, volume
# Pastikan data sudah digabung berdasarkan tanggal

# Muat data
weather_data = pd.read_csv("weather_bakersfield_2023_2024.csv")  # Data cuaca
stock_data = pd.read_csv("tejon_ranch_stock_data_2023_2024.csv")      # Data saham

# Gabungkan data cuaca dan saham berdasarkan tanggal
data = pd.merge(
    df_weather, df_clean_vol, left_on='datetime', right_on='date', how='inner'
)



data['monthly_rainfall'] = data['precip'].rolling(window=30).sum()  # Akumulasi curah hujan bulanan

print(data.columns)

# ---- 1. Korelasi Antar Variabel ---- #
# Buat variabel baru: rata-rata suhu mingguan dan akumulasi curah hujan bulanan
data['week_avg_temp'] = data['temp'].rolling(window=7).mean()  # Rata-rata suhu mingguan
data['monthly_rainfall'] = data['precip'].rolling(window=30).sum()  # Akumulasi curah hujan bulanan

# Normalisasi atau standarisasi variabel numerik
numerical_features = ['temp', 'dew', 'volume', 'week_avg_temp', 'monthly_rainfall']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# ---- 2. Lagged Features ---- #
# Lag data saham (harga penutupan 1-3 hari sebelumnya)
for lag in range(1, 4):
    data[f'close_lag_{lag}'] = data['close'].shift(lag)

# Lag data cuaca (misalnya, suhu dan curah hujan)
for lag in range(1, 4):
    data[f'temp_lag_{lag}'] = data['temp'].shift(lag)
    data[f'precip_lag_{lag}'] = data['precip'].shift(lag)  # Mengganti rain menjadi precip

# ---- 3. Rolling Statistics ---- #
# Tambahkan rata-rata bergerak, standar deviasi, dan median
rolling_window = 7  # Contoh: rolling 7 hari
data['close_ma'] = data['close'].rolling(window=rolling_window).mean()  # Moving average
data['close_std'] = data['close'].rolling(window=rolling_window).std()  # Standar deviasi
data['close_median'] = data['close'].rolling(window=rolling_window).median()  # Median

# Rolling statistics untuk suhu
data['temp_ma'] = data['temp'].rolling(window=rolling_window).mean()  # Moving average suhu
data['temp_std'] = data['temp'].rolling(window=rolling_window).std()  # Standar deviasi suhu

# ---- Drop NaN rows akibat rolling atau lagging ---- #
data = data.dropna()

# ---- Save Processed Data ---- #
data.to_csv("processed_data.csv", index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---- 1. Mengatasi Nilai NaN ---- #
# Periksa apakah ada nilai NaN atau null
print(data.isnull().sum())

# Menghapus baris yang mengandung nilai NaN
data = data.dropna()

#  ---- 2. Encoding Kategorikal ---- #
# Jika ada fitur kategorikal, seperti 'conditions', lakukan encoding
# Contoh: Label Encoding untuk kolom 'conditions'
le = LabelEncoder()
data['conditions_encoded'] = le.fit_transform(data['conditions'])

# Atau bisa menggunakan One-Hot Encoding jika ingin memisahkan setiap kategori cuaca
# One-Hot Encoding untuk kondisi cuaca
# data = pd.get_dummies(data, columns=['conditions'])

# ---- 3. Pembagian Data (Training Set dan Testing Set) ---- #
# Pisahkan data menjadi fitur (X) dan target (y)
# Misalnya, kita ingin memprediksi 'close' berdasarkan fitur lainnya
X = data.drop(columns=['datetime', 'date', 'close', 'conditions'])  # Fitur
y = data['close']  # Target (Harga penutupan)

# Pembagian data menjadi training set dan testing set (70% untuk training, 30% untuk testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- 4. Normalisasi atau Standarisasi ---- #
# Untuk memastikan bahwa data numerik distandarisasi, terutama untuk model berbasis gradient descent (misalnya, regresi atau neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Menstandarisasi data training
X_test = scaler.transform(X_test)  # Menstandarisasi data testing dengan scaler yang sama

# ---- 5. Simpan Dataset yang Sudah Diproses ---- #
# Menyimpan data training dan testing untuk digunakan dalam pemodelan selanjutnya
# Anda bisa menyimpan data dalam file CSV jika diperlukan
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
y_train_df = pd.DataFrame(y_train, columns=['close'])
y_test_df = pd.DataFrame(y_test, columns=['close'])

X_train_df.to_csv("X_train.csv", index=False)
X_test_df.to_csv("X_test.csv", index=False)
y_train_df.to_csv("y_train.csv", index=False)
y_test_df.to_csv("y_test.csv", index=False)

# Menampilkan info hasil encoding dan pembagian data
print("Data training dan testing telah diproses dan disimpan.")

# Import model untuk regresi
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Mempersiapkan data (X_train, X_test, y_train, y_test) sudah diproses sebelumnya

# 2. Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Prediksi dan evaluasi model
y_pred_lr = linear_reg.predict(X_test)
print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# 3. Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Prediksi dan evaluasi model
y_pred_rf = rf_regressor.predict(X_test)
print("\nRandom Forest Regressor:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# 4. Gradient Boosting Regressor (misalnya XGBoost)
from xgboost import XGBRegressor

xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_regressor.fit(X_train, y_train)

# Prediksi dan evaluasi model
y_pred_xgb = xgb_regressor.predict(X_test)
print("\nXGBoost Regressor:")
print("MSE:", mean_squared_error(y_test, y_pred_xgb))
print("R2 Score:", r2_score(y_test, y_pred_xgb))

# Import model untuk klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Mengubah target menjadi biner (naik/turun)
y_class = (data['price_change'] > 0).astype(int)  # 1 jika naik, 0 jika turun

# Pisahkan data menjadi fitur (X) dan target (y_class)
X_class = data.drop(columns=['datetime', 'date', 'close', 'conditions', 'price_change'])  # Fitur
y_class = y_class  # Target (Perubahan harga)

# **Menskala Data**
scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_class)

# Pembagian data menjadi training set dan testing set (70% untuk training, 30% untuk testing)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class_scaled, y_class, test_size=0.3, random_state=42)

# 2. **Logistic Regression**
log_reg = LogisticRegression(random_state=42, max_iter=500)  # Meningkatkan max_iter
log_reg.fit(X_train_class, y_train_class)

# Prediksi dan evaluasi model
y_pred_lr_class = log_reg.predict(X_test_class)
print("Logistic Regression Classification:")
print("Accuracy:", accuracy_score(y_test_class, y_pred_lr_class))
print(classification_report(y_test_class, y_pred_lr_class))

# 3. **Random Forest Classifier**
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_class, y_train_class)

# Prediksi dan evaluasi model
y_pred_rf_class = rf_classifier.predict(X_test_class)
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test_class, y_pred_rf_class))
print(classification_report(y_test_class, y_pred_rf_class))

# 4. **Support Vector Machine (SVM)**
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_class, y_train_class)

# Prediksi dan evaluasi model
y_pred_svm_class = svm_classifier.predict(X_test_class)
print("\nSupport Vector Machine (SVM):")
print("Accuracy:", accuracy_score(y_test_class, y_pred_svm_class))
print(classification_report(y_test_class, y_pred_svm_class))
