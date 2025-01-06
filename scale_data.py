import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

# Membaca dataset
try:
    df = pd.read_csv('data_group_cleaned.csv')
    print("Dataset berhasil dibaca.")
except FileNotFoundError:
    print("Error: File 'data_group_cleaned.csv' tidak ditemukan.")
    exit()

# Menampilkan informasi dataset
print("\nKolom yang tersedia:")
print(df.columns)

print("\nTipe data setiap kolom:")
print(df.dtypes)

print("\nBeberapa baris pertama data:")
print(df.head())

# Mengecek kolom numerik
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print(f"\nKolom numerik yang dipilih: {numeric_columns}")

# Jika tidak ada kolom numerik, mencoba mengekstrak angka dari teks
if len(numeric_columns) == 0:
    print("\nTidak ada kolom numerik yang ditemukan. Mencoba mengekstrak angka dari teks...")

    # Fungsi untuk mengekstrak angka dari teks dan merangkum menjadi satu nilai
    def extract_numbers(text):
        numbers = re.findall(r'\d+', str(text))  # Cari semua angka dalam teks
        if numbers:
            return sum(map(int, numbers))  # Gunakan jumlah semua angka sebagai nilai
        return None

    # Terapkan fungsi pada setiap kolom teks
    for column in df.columns:
        df[f'{column}_numeric'] = df[column].apply(lambda x: extract_numbers(x))

    # Menyimpan kolom dengan data numerik baru
    numeric_columns = [col for col in df.columns if '_numeric' in col]

    if len(numeric_columns) == 0:
        print("Error: Tidak ada angka yang berhasil diekstrak dari teks. Proses scaling dihentikan.")
        exit()

    print(f"\nKolom numerik yang berhasil diekstrak: {numeric_columns}")

# Menyiapkan data untuk scaling
numeric_data = df[numeric_columns].dropna()  # Menghapus baris dengan nilai NaN
if numeric_data.empty:
    print("Error: Tidak ada data numerik yang valid untuk di-scale.")
    exit()

# Melakukan scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Menyimpan hasil scaling
scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)
scaled_df.to_csv('scaled_data.csv', index=False)

print("\nScaling berhasil dilakukan. Hasil:")
print(scaled_df.head())
print("\nData yang sudah di-scale disimpan di 'scaled_data.csv'.")
