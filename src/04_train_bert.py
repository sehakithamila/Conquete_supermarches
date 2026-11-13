from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

GOLD_PRODUCTS = Path("data/gold/products")
MODEL_DIR = Path("models/distilbert_product")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Les ingrédients sont surtout en français -> modèle multilingue
MODEL_CKPT = "distilbert-base-multilingual-cased"
N_SAMPLES = 5000


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def train_bert():
    print("Chargement des données pour le fine-tuning BERT...")
    df = pd.read_parquet(GOLD_PRODUCTS, columns=["ingredients_text", "nova_group"])
    df = df.dropna()
    df = df[df["ingredients_text"].str.len() > 5]

    df_bert = df.rename(columns={"ingredients_text": "text", "nova_group": "label"})
    df_bert["label"] = (df_bert["label"] == 4).astype(int)  # Nova 4 vs autres

    # Échantillon aléatoire (et non .head(), qui serait biaisé par l'ordre du fichier)
    df_bert = df_bert.sample(n=min(N_SAMPLES, len(df_bert)), random_state=42)
    print(f"{len(df_bert):,} exemples | part Nova 4 : {df_bert['label'].mean():.1%}")

    dataset = Dataset.from_pandas(df_bert.reset_index(drop=True))
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(f"Chargement du tokenizer et du modèle {MODEL_CKPT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    def tokenize_function(examples):
        # pas de padding ici : le DataCollator pad dynamiquement par batch (plus rapide sur CPU)
        return tokenizer(examples["text"], truncation=True, max_length=128)

    encoded = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",   # pas de checkpoints intermédiaires (économise le disque)
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        processing_class=tokenizer,  # remplace l'ancien argument `tokenizer=`
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Lancement de l'entraînement DistilBERT...")
    trainer.train()
    print("Évaluation finale :", trainer.evaluate())

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Modèle BERT sauvegardé dans : {MODEL_DIR}")


if __name__ == "__main__":
    train_bert()