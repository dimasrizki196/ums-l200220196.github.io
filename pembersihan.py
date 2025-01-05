# momer 3
import pandas as pd
import re

# Memuat dataset dengan melewatkan baris bermasalah
df = pd.read_csv('data_group.csv', on_bad_lines='skip')

# Lanjutkan dengan pembersihan data seperti sebelumnya
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9,\.!?;:()\-_\s]', '', str(text))

df_cleaned = df.applymap(clean_text)

# Menyimpan hasil pembersihan ke file baru
df_cleaned.to_csv('data_group_cleaned.csv', index=False)

# Menampilkan hasil pembersihan
print(df_cleaned.head())

# with open('data_group.csv', 'r') as file:
#     lines = file.readlines()
#     for i, line in enumerate(lines):
#         columns = line.split(',')
#         if len(columns) != 3:  # Ganti dengan jumlah kolom yang sesuai
#             print(f'Line {i+1} has {len(columns)} columns: {line}')
