from pathlib import Path
import requests

BASE = Path("data")
RAW = BASE / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DEST = RAW / "food.parquet"

HF_URL = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
OFF_URL = "https://static.openfoodfacts.org/data/food.parquet"
USER_AGENT = "SoGood-Student-Project/1.0 (ipssi; codespace)"

def download():
    if DEST.exists() and DEST.stat().st_size > 10 * 1024 * 1024:
        print(f"Déjà présent : {DEST}")
        return
    print("Téléchargement en cours...")
    for url in [HF_URL, OFF_URL]:
        try:
            r = requests.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            with open(DEST, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"Téléchargé avec succès : {DEST}")
            return
        except Exception as e:
            print(f"Échec {url}: {e}")
    raise RuntimeError("Toutes les sources de téléchargement ont échoué.")

if __name__ == "__main__":
    download()