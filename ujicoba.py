from metaflow import FlowSpec, step

class HelloFlow(FlowSpec):

    @step
    def start(self):
        print("Hello, Metaflow!")
        self.next(self.end)

    @step
    def end(self):
        print("Flow Complete!")

if __name__ == "__main__":
    HelloFlow()  # Jangan memanggil langsung di sini

from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import string
import matplotlib.pyplot as plt
import os

class ClusteringFlow(FlowSpec):
    
    num_clusters = Parameter('num_clusters', default=3, type=int, help="Number of clusters")

    @step
    def start(self):
        print("Step start initiated")
        try:
            if not os.path.exists('data_group_cleaned.csv'):
                raise FileNotFoundError("File 'data_group_cleaned.csv' tidak ditemukan!")
            print("File exists")
            self.df = pd.read_csv('data_group_cleaned.csv')
            print(f"Data loaded with shape: {self.df.shape}")
            print(f"Columns in the dataset: {self.df.columns}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

        # Menyiapkan data teks untuk clustering
        self.df_cleaned = self.df.iloc[:, 0].dropna()  # Mengambil hanya kolom pertama dan menghapus NaN
        print("Data cleaned and ready for processing")

        # Melakukan preprocessing teks
        self.df_cleaned = self.df_cleaned.apply(self.preprocess_text)

        # Menggunakan TF-IDF untuk mengubah teks menjadi representasi numerik
        vectorizer = TfidfVectorizer(stop_words='english')
        self.X = vectorizer.fit_transform(self.df_cleaned)

        print(f"TF-IDF matrix shape: {self.X.shape}")
        
        self.next(self.cluster)

    def preprocess_text(self, text):
        # Fungsi untuk membersihkan teks
        text = text.lower()  # Mengonversi ke huruf kecil
        text = ''.join([char for char in text if char not in string.punctuation])  # Menghapus tanda baca
        return text

    @step
    def cluster(self):
        try:
            print(f"Clustering with {self.num_clusters} clusters")
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            self.clusters = kmeans.fit_predict(self.X)
            self.centroids = kmeans.cluster_centers_
            self.labels = kmeans.labels_
        except Exception as e:
            print(f"Error during clustering: {e}")
            raise
        self.next(self.analyze)

    @step
    def analyze(self):
        print(f"Cluster labels: {self.labels}")
        print(f"Cluster centroids: \n{self.centroids}")

        # Menambahkan label cluster ke dataframe dan menampilkan 3 data teratas di setiap cluster
        df_with_clusters = self.df.copy()
        df_with_clusters['Cluster'] = self.labels

        # Menampilkan 3 baris pertama dari setiap cluster
        top_entries_per_cluster = df_with_clusters.groupby('Cluster').head(3)
        print("Top 3 entries from each cluster:")
        print(top_entries_per_cluster)

        # Menyimpan hasil clustering ke file CSV
        df_with_clusters.to_csv('data_with_clusters.csv', index=False)

        # Jika jumlah fitur lebih dari dua, gunakan PCA untuk reduksi dimensi
        if self.X.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(self.X.toarray())  # Konversi sparse matrix ke dense jika perlu
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.labels, cmap='viridis')
            plt.title("Clustering Results (PCA Reduced)")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.colorbar(label='Cluster')
            plt.show()
        else:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels, cmap='viridis')
            plt.title("Clustering Results")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.colorbar(label='Cluster')
            plt.show()

        self.next(self.end)

    @step
    def end(self):
        print("Clustering Flow Complete!")

if __name__ == "__main__":
    ClusteringFlow()

