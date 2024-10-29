from metaflow import FlowSpec, step

class KuliahFlow(FlowSpec):

    @step
    def start(self):
        # Daftar mahasiswa yang mengikuti proses perkuliahan
        self.mahasiswas = ['Dimas', 'Rani', 'Budi', 'Siti', 'Andi']
        self.next(self.bayar_spp, foreach='mahasiswas')

    @step
    def bayar_spp(self):
        self.nama = self.input
        print(f"{self.nama} membayar SPP.")
        self.next(self.isi_krs)

    @step
    def isi_krs(self):
        print(f"{self.nama} mengisi KRS.")
        self.next(self.mulai_kuliah)

    @step
    def mulai_kuliah(self):
        print(f"{self.nama} mulai mengikuti perkuliahan.")
        self.next(self.ikut_ujian)

    @step
    def ikut_ujian(self):
        print(f"{self.nama} mengikuti ujian.")
        # Contoh nilai acak untuk setiap mahasiswa
        self.nilai_akhir = 85  # Misalnya nilai ujian akhir
        self.next(self.join)

    @step
    def join(self, inputs):
        # Menghitung total nilai akhir dari setiap mahasiswa
        total_nilai = 0
        count = 0
        
        for inp in inputs:
            total_nilai += inp.nilai_akhir
            count += 1
        
        # Menghitung rata-rata nilai akhir
        if count > 0:
            self.rata_rata = total_nilai / count
        else:
            self.rata_rata = 0
        
        self.total_nilai = total_nilai
        
        # Melanjutkan ke langkah berikutnya
        self.next(self.end)

    @step
    def end(self):
        print(f"Rata-rata nilai akhir semua mahasiswa adalah {self.rata_rata}")

if __name__ == '__main__':
    KuliahFlow()