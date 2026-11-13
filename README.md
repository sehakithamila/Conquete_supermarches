# 🥗 SoGood — Conquête des supermarchés

Analyse de la qualité nutritionnelle des produits de supermarché à partir de la base **[Open Food Facts](https://world.openfoodfacts.org/)**.

Le projet couvre toute la chaîne : téléchargement des données brutes, nettoyage avec **Apache Spark** (architecture bronze / silver / gold), entraînement de deux modèles (**Random Forest** et **DistilBERT**) et application web **Streamlit** pour explorer les produits et faire des prédictions.

> Projet réalisé dans le cadre de la formation IPSSI.

---

## Fonctionnalités

- Téléchargement sécurisé du dataset Open Food Facts (~7,5 Go, ~4,9 millions de produits) avec vérification d'intégrité.
- ETL Spark en trois couches (bronze → silver → gold), filtré par défaut sur les produits vendus en **France**.
- Score de **controverse** par produit et par marque.
- **Random Forest** : estimation du Nutri-Score (A → E) à partir des valeurs nutritionnelles.
- **DistilBERT** : estimation de la probabilité qu'un produit soit **ultra-transformé (NOVA 4)** à partir de sa liste d'ingrédients.
- Application web interactive : tableaux de bord, recherche, fiche produit, prédictions, classement des marques.

## Architecture du projet

```
Conquete_supermarches/
├── app.py                      # Application Streamlit
├── .streamlit/
│   └── config.toml             # Thème de l'application
├── src/
│   ├── 01_download.py          # Téléchargement du dataset brut
│   ├── 02_etl_spark.py         # ETL Spark : bronze → silver → gold
│   ├── 03_train_rf.py          # Entraînement du Random Forest
│   └── 04_train_bert.py        # Fine-tuning de DistilBERT
├── data/                       # Généré par le pipeline (non versionné)
│   ├── raw/food.parquet
│   ├── bronze/
│   ├── silver/
│   └── gold/
│       ├── products/
│       └── brands/
├── models/                     # Généré par les scripts d'entraînement
│   ├── random_forest_nutriscore.joblib
│   ├── label_encoder.joblib
│   └── distilbert_product/
└── README.md
```

## Installation

### Prérequis

- **Python** 3.11 ou 3.12 recommandé (3.14 fonctionne pour Spark, mais `torch` / `transformers` peuvent ne pas encore le supporter).
- **Java 17+** (requis par Spark ; déjà présent dans les Codespaces).
- **RAM** : 8 Go minimum pour l'ETL (16 Go confortable). **Disque** : ~20 Go libres (fichier brut + couches intermédiaires).

### Dépendances

```bash
pip install -U pyspark pyarrow "pandas<3" numpy requests \
                scikit-learn joblib \
                torch transformers datasets accelerate \
                streamlit plotly
```

## Utilisation

Depuis la racine du projet, dans l'ordre :

```bash
# 1. Télécharger les données (~7,5 Go)
python src/01_download.py

# 2. ETL Spark → data/gold/
python src/02_etl_spark.py

# 3. Entraîner le Random Forest → models/random_forest_nutriscore.joblib
python src/03_train_rf.py

# 4. Fine-tuner DistilBERT → models/distilbert_product/
python src/04_train_bert.py

# 5. Lancer l'application
streamlit run app.py
```

L'app fonctionne dès que l'étape 2 est terminée : les onglets de prédiction signalent simplement l'absence d'un modèle non entraîné.

### Options de l'ETL -- utilisation sur codespaces

Variables d'environnement facultatives pour `02_etl_spark.py` :

| Variable | Effet | Exemple |
|---|---|---|
| `MAX_ROWS` | Ne traite qu'un échantillon (test rapide du pipeline) | `MAX_ROWS=200000` |
| `SPARK_DRIVER_MEMORY` | Force la mémoire du driver Spark | `SPARK_DRIVER_MEMORY=3g` |
| `SPARK_CORES` | Force le nombre de cœurs utilisés | `SPARK_CORES=1` |

```bash
MAX_ROWS=200000 python src/02_etl_spark.py
```

Sans ces variables, le script détecte la RAM disponible et se configure automatiquement. Le pays filtré se change via la variable `COUNTRY` dans le script (`'fr'` par défaut ; `None` pour tout garder).

## Détail du pipeline

| Couche | Contenu |
|---|---|
| **Raw** | Fichier Parquet Open Food Facts tel que téléchargé. |
| **Bronze** | Sélection des colonnes utiles, extraction des nutriments depuis les tableaux imbriqués, texte multilingue (priorité fr → en → main → premier disponible). |
| **Silver** | Filtre pays, dédoublonnage sur `code`, suppression des noms vides, valeurs nutritionnelles aberrantes mises à `NULL` (hors 0–1000), Nutri-Score restreint à `a`–`e`, indicateur `is_food`. |
| **Gold** | Table `products` finale + `controversy_score` ; table `brands` agrégée par marque (≥ 10 produits). |

**Score de controverse** d'un produit :

```
nombre d'additifs  +  5 si Nutri-Score D ou E  +  3 si NOVA 4
```

## Modèles

### Random Forest — Nutri-Score

- **Cible** : `nutriscore_grade` (A, B, C, D, E).
- **Features** : énergie, sucres, sel, graisses, graisses saturées, protéines, fibres, nombre d'additifs, groupe NOVA.
- **Configuration** : 100 arbres, profondeur max 15, split stratifié 80/20.
- Le rapport de classification s'affiche à la fin de l'entraînement.

### DistilBERT — ultra-transformation

- **Cible** : classification binaire, NOVA 4 (ultra-transformé) contre les autres groupes.
- **Entrée** : liste d'ingrédients (tronquée à 128 tokens).
- **Modèle de base** : `distilbert-base-multilingual-cased` (les ingrédients sont majoritairement en français).
- **Entraînement par défaut** : 5 000 exemples, 1 epoch, pour rester rapide sur CPU. Augmente `N_SAMPLES` et `num_train_epochs` dans `04_train_bert.py` pour de meilleurs résultats.

> ⚠️ Le modèle BERT **ne prédit pas le Nutri-Score** : il estime uniquement la probabilité d'être NOVA 4.

## Application Streamlit

| Onglet | Contenu |
|---|---|
| 📊 **Vue d'ensemble** | Indicateurs clés, distributions Nutri-Score et NOVA, heatmap Nutri-Score × NOVA, nuage sucres / sel. |
| 🔎 **Explorer** | Recherche par produit ou marque, tableau filtrable, fiche produit détaillée avec écart à la médiane de la catégorie. |
| 🤖 **Prédire** | Estimation du Nutri-Score (Random Forest) et de la probabilité NOVA 4 (DistilBERT). |
| 🏷️ **Marques** | Classement des marques les plus / moins controversées, avec seuil minimum de produits. |

La barre latérale permet de filtrer par catégorie, groupe NOVA, nombre d'additifs et produits alimentaires. Elle indique aussi quels modèles sont disponibles.

Dans un Codespace, ouvre l'onglet **Ports** pour accéder à l'application une fois lancée.

## Dépannage

**`Job cancelled because SparkContext was shut down` pendant l'ETL**
Spark a été tué par manque de mémoire (OOM). Vérifie la RAM avec `free -h`, ferme les processus inutiles et relance avec une config plus prudente :

```bash
SPARK_DRIVER_MEMORY=3g SPARK_CORES=1 python src/02_etl_spark.py
```

Sinon, passe le Codespace sur une machine 16 Go.

**Le fichier `food.parquet` semble corrompu**
Vérifie qu'il est lisible :

```bash
python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('data/raw/food.parquet').metadata.num_rows)"
```

En cas d'erreur, supprime le fichier et relance `01_download.py`.

**`Données gold introuvables` dans l'app**
Lance l'ETL (`python src/02_etl_spark.py`) depuis la racine du projet, puis `streamlit run app.py` depuis cette même racine.

**Erreur `tokenizer` ou `width="stretch"`**
Mets à jour les bibliothèques : `pip install -U transformers streamlit`.

## Données et licence

Les données proviennent d'[Open Food Facts](https://world.openfoodfacts.org/) et sont distribuées sous licence [ODbL](https://opendatacommons.org/licenses/odbl/1-0/). Toute réutilisation doit citer la source et partager les bases dérivées sous la même licence.

Les scores Nutri-Score et NOVA utilisés ici sont ceux renseignés dans Open Food Facts ; les prédictions des modèles sont **indicatives** et ne remplacent pas un calcul officiel.