import pandas as pd
from sklearn.preprocessing import StandardScaler

# Membaca dataset yang telah dibersihkan
df = pd.read_csv('data_group_cleaned.csv')

# Memeriksa kolom numerik yang ada di dataset
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print(f"Kolom numerik yang dipilih: {numeric_columns}")

# Pastikan ada kolom numerik sebelum melanjutkan
if len(numeric_columns) > 0:
    # Menstandarkan data (scaling)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_columns])

    # Menyimpan data yang sudah diskalakan
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)
    scaled_df.to_csv('scaled_data.csv', index=False)

    print(scaled_df.head())
else:
    print("Tidak ada kolom numerik yang ditemukan untuk di-scale.")
