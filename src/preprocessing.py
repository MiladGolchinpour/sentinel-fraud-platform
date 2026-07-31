from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.features import build_features

PROCESSED_DATA_PATH = Path("data/processed/paysim_features.parquet")
DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud",]

def build_pipline(model) -> Pipeline:
    """Create the preprocessing + model pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["type"],
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline

def prepare_data(csv_path: str | Path):
    """Read the raw dataset, build features, and save as parquet."""

    df = pd.read_csv(csv_path)
    df = build_features(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)

def load_data(csv_path: str | Path):
    """Prepare data, split into train/dev/test, and return datasets."""
    # prepare raw data
    prepare_data(csv_path)

    # load processed file
    dataset = pd.read_parquet(PROCESSED_DATA_PATH)

    # removing unecessary features
    dataset = dataset.drop(columns=DROP_COLUMNS)

    # choosing X, y
    X = dataset.drop(columns=["isFraud"])
    y = dataset["isFraud"]

    # split train/dev/test -> 80/10/10
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    for name, target in [("Train", y_train),("Dev", y_dev),("Test", y_test)]:
        print(f"{name}: ")
        print(f"{len(target):,} samples | ", end="")
        print(f"Fraud = {target.mean()*100:.3f}%")

    print("\nData Loaded Successfully!\n")

    return X_train, X_test, X_dev, y_train, y_test, y_dev
