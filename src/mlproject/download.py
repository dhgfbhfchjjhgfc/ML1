"""Загрузка полного датасета Credit Card Fraud (OpenML id 1597) и создание sample.csv."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

def download() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "creditcard.csv"

    print("Загрузка CreditCardFraudDetection с OpenML (ID 1597)...")
    data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = data.frame

    # Приводим типы: Time, Amount, V1-V28 числовые, Class - 0 или 1
    df["Class"] = df["Class"].astype(int)
    for col in df.columns:
        if col != "Class":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(csv_path, index=False)
    print(f"Полный датасет сохранен в {csv_path} ({len(df)} строк)")

    # Создаем sample.csv: стратифицированная выборка на 2000 строк (с достаточным числом фрода)
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0].sample(n=2000 - len(fraud), random_state=42)
    sample = pd.concat([normal, fraud]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    Path("data").mkdir(exist_ok=True)
    sample.to_csv("data/sample.csv", index=False)
    print(f"data/sample.csv создан: {len(sample)} строк, {sample['Class'].sum()} фрод-транзакций.")

if __name__ == "__main__":
    download()
