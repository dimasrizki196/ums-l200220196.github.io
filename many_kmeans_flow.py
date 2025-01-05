from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import string
import matplotlib.pyplot as plt
import os

# nomer 4
class ClusteringFlow(FlowSpec):
    
    # Parameter untuk jumlah cluster yang diinginkan
    num_clusters = Parameter('num_clusters', default=3, type=int, help="Number of clusters")

    @step
    def start(self):
        print("Step start initiated")
        try:
            # Memastikan file CSV tersedia
            if not os.path.exists('data_group_cleaned.csv'):
                raise FileNotFoundError("File 'data_group_cleaned.csv' tidak ditemukan!")
            print("File exists")
            self.df = pd.read_csv('data_group_cleaned.csv')
            print(f"Data loaded with shape: {self.df.shape}")
            print(f"Columns in the dataset: {self.df.columns}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

        # Menyiapkan data teks untuk clustering (kolom pertama dan menghapus NaN)
        # Pastikan kolom yang dipilih berisi teks yang relevan
        self.df_cleaned = self.df['14/11/23 10.04 - Pesan dan panggilan dienkripsi secara end-to-end. Tidak seorang pun di luar chat ini'].dropna()  
        print("Data cleaned and ready for processing")

        # Preprocessing teks: menghapus tanda baca dan membuat teks menjadi huruf kecil
        self.df_cleaned = self.df_cleaned.apply(self.preprocess_text)

        # Menggunakan TF-IDF untuk mengubah teks menjadi representasi numerik
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(self.df_cleaned)

        print(f"TF-IDF matrix shape: {self.X.shape}")
        
        # Melanjutkan ke langkah cluster
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
            # Melakukan clustering dengan KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            self.clusters = kmeans.fit_predict(self.X)
            self.centroids = kmeans.cluster_centers_
            self.labels = kmeans.labels_
        except Exception as e:
            print(f"Error during clustering: {e}")
            raise

        # Melanjutkan ke langkah selanjutnya
        self.next(self.detail)

    @step
    def detail(self):
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

        self.next(self.analyze)

# nomer 5
    @step
    def analyze(self):
        # Menampilkan hasil label cluster dan centroid untuk berbagai jumlah cluster
        for clusters in [3, 4, 5]:
            print(f"Clustering with {clusters} clusters")
            
            # Melakukan clustering dengan KMeans untuk jumlah cluster yang berbeda
            kmeans = KMeans(n_clusters=clusters, random_state=42)
            labels = kmeans.fit_predict(self.X)
            centroids = kmeans.cluster_centers_

            # Menambahkan label cluster ke dataframe
            df_with_clusters = self.df.copy()
            df_with_clusters['Cluster'] = labels

            # Menampilkan 3 data teratas dari setiap cluster
            top_entries_per_cluster = df_with_clusters.groupby('Cluster').head(3)
            print(f"Top 3 entries from each cluster for {clusters} clusters:")
            print(top_entries_per_cluster)

            # Menyimpan hasil clustering ke file CSV
            df_with_clusters.to_csv(f'data_with_clusters_{clusters}.csv', index=False)
            
            # nomer 6
            # Menampilkan 3 kata teratas dari setiap cluster berdasarkan bobot tertinggi pada pusat cluster
            feature_names = self.vectorizer.get_feature_names_out()
            for i in range(clusters):
                cluster_center = centroids[i]
                top_indices = cluster_center.argsort()[-3:][::-1]  # Indeks 3 kata dengan bobot tertinggi
                top_words = [feature_names[index] for index in top_indices]
                print(f"Cluster {i} top 3 words: {', '.join(top_words)}")

            # Jika jumlah fitur lebih dari dua, lakukan reduksi dimensi dengan PCA
            if self.X.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(self.X.toarray())  # Konversi sparse matrix ke dense jika perlu
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
                plt.title(f"Clustering Results (PCA Reduced) with {clusters} clusters")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                plt.colorbar(label='Cluster')
                plt.show()
            else:
                plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis')
                plt.title(f"Clustering Results with {clusters} clusters")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.colorbar(label='Cluster')
                plt.show()

        # Menambahkan transisi ke langkah 'end'
        self.next(self.end)


    @step
    def end(self):
        print("Clustering Flow Complete!")

# Menjalankan Flow
if __name__ == "__main__":
    ClusteringFlow()
