import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


def load_bank_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # En Bank Marketing, 'unknown' representa información faltante/desconocida en categóricas
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.drop(TARGET)
    df[cat_cols] = df[cat_cols].replace("unknown", pd.NA)
    return df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor

# Agrega esta función arriba de preprocess_and_split() en src/preprocess.py
def get_raw_splits(df: pd.DataFrame | None = None):
    """Divide los datos crudos en 70/15/15 estratificado excluyendo 'duration' para evitar fuga temporal."""
    if df is None:
        df = load_bank_data()

    cols_to_drop = [TARGET, "duration"] if "duration" in df.columns else [TARGET]
    X = df.drop(columns=cols_to_drop)
    y = (df[TARGET] == "yes").astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_and_split(df: pd.DataFrame | None = None):
    if df is None:
        df = load_bank_data()

    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "yes").astype(int)

    # Primero separamos test (15%) de forma estratificada
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Luego separamos validación (15% del total -> VAL_SIZE / (1 - TEST_SIZE) de temp)
    val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # El preprocesador se AJUSTA (fit) ÚNICAMENTE con X_train para evitar fuga de datos
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    X_train_df.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_val_df.to_csv(PROCESSED_DATA_DIR / "X_val.csv", index=False)
    X_test_df.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False, header=True)
    y_val.to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False, header=True)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False, header=True)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    print(f"Train: {X_train_df.shape} | Val: {X_val_df.shape} | Test: {X_test_df.shape}")
    print(f"Proporción clase positiva (y=1) -> Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    preprocess_and_split()