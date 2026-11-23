from pathlib import Path

# Raíz del proyecto: <Proyecto_Data_Science>
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_NAME = "bank-full.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME

TARGET = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # 70% train / 15% val / 15% test estratificado

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)