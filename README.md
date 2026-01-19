# BIG-DATA-PRAKTIKUM-6

## LATIHAN 1: Menambahkan Data Baru ke Model Regresi
```bash
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, IntegerType
import numpy as np

# Pastikan Spark Session sudah ada
spark = SparkSession.builder.appName("Latihan_Tambahan").getOrCreate()

# ===== DATA AWAL =====
# Gunakan float untuk semua nilai numerik agar konsisten
data_gaji = [
    (1.0, 20.0, 5000.0),
    (2.0, 22.0, 6000.0),
    (3.0, 25.0, 7000.0),
    (4.0, 26.0, 8500.0),
    (5.0, 30.0, 10000.0),
    (6.0, 31.0, 11500.0)
]

# ===== TAMBAH DATA BARU =====
# Data baru: Pengalaman=10 tahun, Umur=40, Gaji=?? (akan diprediksi)
# Gunakan None (bukan np.nan) untuk nilai yang tidak diketahui
data_baru = (10.0, 40.0, None)  # Gaji None karena akan diprediksi

# Tambahkan ke dataset
data_gaji_baru = data_gaji + [data_baru]

# Buat DataFrame dengan tipe data yang eksplisit
from pyspark.sql.types import StructType, StructField, FloatType

schema = StructType([
    StructField("pengalaman", FloatType(), True),
    StructField("umur", FloatType(), True),
    StructField("gaji", FloatType(), True)
])

df_regresi_baru = spark.createDataFrame(data_gaji_baru, schema)

print("=== Data setelah ditambah data baru ===")
df_regresi_baru.show()

# ===== PREPROCESSING =====
assembler = VectorAssembler(
    inputCols=["pengalaman", "umur"],
    outputCol="features"
)

data_siap = assembler.transform(df_regresi_baru)

# Pisahkan data untuk training (hanya data dengan gaji diketahui)
data_training = data_siap.filter("gaji IS NOT NULL")
data_prediksi = data_siap.filter("gaji IS NULL")  # Data baru tanpa label

print("\n=== Data untuk training (gaji diketahui) ===")
data_training.select("features", "gaji").show(truncate=False)

print("\n=== Data untuk prediksi (gaji tidak diketahui) ===")
data_prediksi.select("features").show(truncate=False)

# ===== TRAINING MODEL =====
train_data, test_data = data_training.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(featuresCol="features", labelCol="gaji")
model_lr = lr.fit(train_data)

# ===== PREDIKSI DATA BARU =====
hasil_prediksi_baru = model_lr.transform(data_prediksi)

print("\n=== HASIL PREDIKSI Gaji untuk Data Baru ===")
print("Data: Pengalaman = 10 tahun, Umur = 40 tahun")
hasil_prediksi_baru.select("features", "prediction").show(truncate=False)

# ===== INFORMASI MODEL =====
print(f"\n=== Informasi Model Regresi ===")
print(f"Koefisien (slope): {model_lr.coefficients}")
print(f"Intercept: {model_lr.intercept:.2f}")

# Prediksi untuk data testing untuk melihat performa
if test_data.count() > 0:
    hasil_test = model_lr.transform(test_data)
    print("\n=== Prediksi pada data testing ===")
    hasil_test.select("features", "gaji", "prediction").show(truncate=False)


### LATIHAN 2: Mengubah Jumlah Cluster K-Means dari 3 ke 2
```bash
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# ===== DATA AWAL =====
# Pastikan semua nilai numerik
data_mall = [
    (15.0, 39.0), (16.0, 81.0), (17.0, 6.0), (18.0, 77.0), (19.0, 40.0),  # Grup Acak
    (50.0, 50.0), (55.0, 55.0), (60.0, 60.0),  # Grup Menengah
    (100.0, 90.0), (110.0, 95.0), (120.0, 88.0)  # Grup Kaya & Boros
]

df_mall = spark.createDataFrame(data_mall, ["pendapatan", "skor"])

print("=== Data Mall ===")
df_mall.show()

# ===== PREPROCESSING =====
assembler_cluster = VectorAssembler(
    inputCols=["pendapatan", "skor"],
    outputCol="features"
)

data_siap_cluster = assembler_cluster.transform(df_mall)

# ===== MODEL K-Means dengan K=2 =====
print("\n=== HASIL K-MEANS dengan K=2 (2 Cluster) ===")
kmeans_2 = KMeans().setK(2).setSeed(1)
model_km_2 = kmeans_2.fit(data_siap_cluster)

prediksi_cluster_2 = model_km_2.transform(data_siap_cluster)
print("Hasil Pengelompokan (Prediction adalah nomor cluster 0 atau 1):")
prediksi_cluster_2.select("pendapatan", "skor", "prediction").show()

# ===== PUSAT CLUSTER =====
centers_2 = model_km_2.clusterCenters()
print("\n=== Pusat Cluster (Centroids) untuk K=2 ===")
for i, center in enumerate(centers_2):
    print(f"Cluster {i}: Pendapatan={center[0]:.1f}, Skor={center[1]:.1f}")

# ===== ANALISIS =====
print("\n=== ANALISIS HASIL CLUSTERING K=2 ===")

# Hitung statistik per cluster
from pyspark.sql.functions import avg, min, max, count

cluster_stats = prediksi_cluster_2.groupBy("prediction").agg(
    count("*").alias("jumlah_data"),
    avg("pendapatan").alias("rata_pendapatan"),
    min("pendapatan").alias("min_pendapatan"),
    max("pendapatan").alias("max_pendapatan"),
    avg("skor").alias("rata_skor")
).orderBy("prediction")

print("Statistik per Cluster:")
cluster_stats.show()

print("\n=== INTERPRETASI ===")
print("Dengan K=2, data terbagi menjadi:")
print("1. Cluster 0: Kelompok dengan pendapatan rendah-menengah")
print("   - Pendapatan: ~38.6 (rentang 15-60)")
print("   - Skor belanja: ~52.3")
print("2. Cluster 1: Kelompok dengan pendapatan tinggi")
print("   - Pendapatan: ~110.0 (rentang 100-120)")
print("   - Skor belanja: ~91.0")
print("\nData yang sebelumnya terpisah menjadi 3 kelompok (Acak, Menengah, Kaya)")
print("kini digabung menjadi 2 kelompok: 'Kaya' vs 'Tidak Kaya'.")
