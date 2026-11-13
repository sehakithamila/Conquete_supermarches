import os
from pathlib import Path

import pyarrow.dataset as ds
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. Chemins
BASE = Path("data")
RAW = BASE / "raw"
BRONZE = BASE / "bronze"
SILVER = BASE / "silver"
GOLD = BASE / "gold"
SPARK_TMP = BASE / "spark_tmp"

for p in (BRONZE, SILVER, GOLD, SPARK_TMP):
    p.mkdir(parents=True, exist_ok=True)


# 2. Mémoire disponible -> config Spark adaptée (évite l'OOM killer)
def available_ram_gb() -> float:
    """RAM réellement disponible (limite cgroup du conteneur, sinon RAM physique)."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v.isdigit() and int(v) < (1 << 50):
                return int(v) / 2**30
        except OSError:
            pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


ram = available_ram_gb()
# On laisse ~40 % de la RAM à l'OS, à Python et à la mémoire off-heap de la JVM
driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", f"{max(1, int(ram * 0.6))}g")
cores = os.environ.get("SPARK_CORES", "2" if ram < 12 else "4")
print(f"RAM détectée: {ram:.1f} Go -> driver.memory={driver_mem}, cores={cores}")

spark = (
    SparkSession.builder.appName("sogood")
    .master(f"local[{cores}]")
    .config("spark.driver.memory", driver_mem)
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.local.dir", str(SPARK_TMP))
    .config("spark.sql.shuffle.partitions", "16")
    # petites partitions + petits batchs : colonnes imbriquées très lourdes en mémoire
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
    .config("spark.sql.ansi.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version, "- ANSI mode:", spark.conf.get("spark.sql.ansi.enabled"))

# 3. Chargement des données brutes
raw_path = str(RAW / "food.parquet")
print(f"Chargement des données depuis : {raw_path}")
raw = spark.read.parquet(raw_path)

# Optionnel : tester vite sur un échantillon -> MAX_ROWS=200000 python src/02_etl_spark.py
max_rows = os.environ.get("MAX_ROWS")
if max_rows:
    raw = raw.limit(int(max_rows))
    print(f"Mode échantillon : {int(max_rows):,} lignes max")


def nutri(name):
    return F.expr(
        f"try_element_at(filter(nutriments, x -> x.name = '{name}'), 1).`100g`"
    ).cast("double")


def multilang(col):
    return F.coalesce(
        F.expr(f"try_element_at(filter({col}, x -> x.lang = 'fr'), 1).text"),
        F.expr(f"try_element_at(filter({col}, x -> x.lang = 'en'), 1).text"),
        F.expr(f"try_element_at(filter({col}, x -> x.lang = 'main'), 1).text"),
        F.expr(f"try_element_at({col}, 1).text"),
    )


# --- BRONZE ---
bronze = raw.select(
    F.col("code"),
    multilang("product_name").alias("product_name"),
    multilang("ingredients_text").alias("ingredients_text"),
    F.col("brands"),
    F.col("categories"),
    F.expr("try_element_at(categories_tags, 1)").alias("main_category"),
    F.concat_ws(",", F.col("countries_tags")).alias("countries_str"),
    F.lower(F.trim(F.col("nutriscore_grade"))).alias("nutriscore_grade"),
    F.col("nutriscore_score").cast("double").alias("nutriscore_score"),
    F.col("nova_group").cast("int").alias("nova_group"),
    F.col("additives_n").cast("int").alias("additives_n"),
    F.when(F.col("additives_tags").isNotNull(), F.size(F.col("additives_tags")))
    .otherwise(F.lit(0)).alias("additives_count"),
    nutri("energy-kcal").alias("energy_kcal_100g"),
    nutri("fat").alias("fat_100g"),
    nutri("saturated-fat").alias("saturated_fat_100g"),
    nutri("carbohydrates").alias("carbohydrates_100g"),
    nutri("sugars").alias("sugars_100g"),
    nutri("fiber").alias("fiber_100g"),
    nutri("proteins").alias("proteins_100g"),
    nutri("salt").alias("salt_100g"),
    nutri("sodium").alias("sodium_100g"),
)

# On écrit d'abord (une seule lecture du gros fichier), on compte ensuite sur le résultat
bronze.write.mode("overwrite").parquet(str(BRONZE))
bronze_df = spark.read.parquet(str(BRONZE))
print(f"Lignes bronze: {bronze_df.count():,}")
print("Colonnes bronze:", bronze_df.columns)
print("OK bronze écrit")

# Aperçu SANS charger toute la table en mémoire
preview = ds.dataset(str(BRONZE), format="parquet").head(3).to_pandas()
print("\nAperçu:")
print(preview[["code", "product_name", "brands", "nutriscore_grade",
               "energy_kcal_100g", "sugars_100g", "additives_count"]].to_string())

# --- SILVER ---
COUNTRY = "fr"

silver = bronze_df
if COUNTRY:
    needle = {"fr": "en:france", "us": "en:united-states", "es": "en:spain",
              "it": "en:italy", "de": "en:germany", "uk": "en:united-kingdom"
              }.get(COUNTRY, f"en:{COUNTRY}")
    silver = silver.where(F.col("countries_str").contains(needle))

silver = silver.dropDuplicates(["code"])
silver = silver.where(F.col("product_name").isNotNull() & (F.length("product_name") >= 2))

for c in ["energy_kcal_100g", "fat_100g", "saturated_fat_100g", "carbohydrates_100g",
          "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g"]:
    silver = silver.withColumn(c, F.when((F.col(c) >= 0) & (F.col(c) <= 1000), F.col(c)))

# 'unknown' / 'not-applicable' ne sont pas de vrais grades -> NULL
silver = silver.withColumn(
    "nutriscore_grade",
    F.when(F.col("nutriscore_grade").isin("a", "b", "c", "d", "e"), F.col("nutriscore_grade")),
)

silver = silver.withColumn(
    "is_food",
    F.when(F.lower(F.col("main_category")).startswith("en:non-food"), F.lit(False))
    .otherwise(F.lit(True)),
)

silver.write.mode("overwrite").parquet(str(SILVER))
silver_df = spark.read.parquet(str(SILVER))
print(f"Silver final: {silver_df.count():,} lignes")
print(f"Dont Nutri-Score renseigné: {silver_df.where(F.col('nutriscore_grade').isNotNull()).count():,}")
print("OK silver")

# --- GOLD ---
products = silver_df.select(
    "code", "product_name", "brands", "categories", "main_category",
    "nutriscore_grade", "nutriscore_score", "nova_group",
    "additives_count",
    "energy_kcal_100g", "sugars_100g", "salt_100g", "fat_100g",
    F.col("saturated_fat_100g").alias("saturated-fat_100g"),
    "proteins_100g", "fiber_100g", "is_food", "ingredients_text",
)

products = products.withColumn(
    "controversy_score",
    F.coalesce(F.col("additives_count"), F.lit(0)).cast("double") * 1.0
    + F.when(F.col("nutriscore_grade").isin("d", "e"), F.lit(5.0)).otherwise(F.lit(0.0))
    + F.when(F.col("nova_group") == 4, F.lit(3.0)).otherwise(F.lit(0.0)),
)

brand_agg = (
    products.where(F.col("brands").isNotNull() & (F.col("brands") != ""))
    .groupBy(F.lower(F.trim(F.split(F.col("brands"), ",").getItem(0))).alias("brand"))
    .agg(F.count("*").alias("n_products"),
         F.avg("controversy_score").alias("avg_controversy"),
         F.avg("additives_count").alias("avg_additives"),
         F.avg("sugars_100g").alias("avg_sugars"),
         F.avg("salt_100g").alias("avg_salt"))
    .where(F.col("n_products") >= 10)
)

products.write.mode("overwrite").parquet(str(GOLD / "products"))
brand_agg.write.mode("overwrite").parquet(str(GOLD / "brands"))

n_products = spark.read.parquet(str(GOLD / "products")).count()
n_brands = spark.read.parquet(str(GOLD / "brands")).count()
print(f"OK gold - Produits: {n_products:,}, marques: {n_brands:,}")

spark.stop()