import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from config import FIGURES_DIR, RAW_DATA_PATH, TARGET


def run_eda():
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    print("=== RESUMEN GENERAL DEL DATASET ===")
    print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    print("\nDistribución de la variable objetivo (y):")
    counts = df[TARGET].value_counts()
    props = df[TARGET].value_counts(normalize=True) * 100
    for cls in counts.index:
        print(f"  {cls}: {counts[cls]} ({props[cls]:.2f}%)")

    # Conteo de 'unknown' en columnas categóricas
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.drop(TARGET)
    unknown_counts = (df[cat_cols] == "unknown").sum()
    unknown_counts = unknown_counts[unknown_counts > 0].sort_values(ascending=False)
    print("\nColumnas con valores 'unknown':")
    for col, val in unknown_counts.items():
        print(f"  {col}: {val} ({val / len(df) * 100:.2f}%)")

    # ---------------------------------------------------------
    # GRÁFICA 1: Distribución de la variable objetivo (Desbalance)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette=["#4C72B0", "#DD8452"], legend=False, ax=ax)
    ax.set_title("Distribución de la Variable Objetivo (Suscripción a Depósito)")
    ax.set_xlabel("¿Suscribió depósito a plazo? (y)")
    ax.set_ylabel("Número de clientes")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,}\n({height / len(df) * 100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_distribucion_target.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRÁFICA 2: Valores 'unknown' por variable categórica
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=unknown_counts.values, y=unknown_counts.index, hue=unknown_counts.index, palette="viridis", legend=False, ax=ax)
    ax.set_title("Cantidad de Registros 'unknown' por Variable Categórica")
    ax.set_xlabel("Número de registros 'unknown'")
    ax.set_ylabel("Variable")
    for i, val in enumerate(unknown_counts.values):
        ax.text(val + 300, i, f"{val:,} ({val / len(df) * 100:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, unknown_counts.max() * 1.22)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_valores_unknown.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRÁFICA 3: Distribución de variables numéricas clave por clase
    # ---------------------------------------------------------
    num_vars = ["age", "balance", "duration", "campaign"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flat, num_vars):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=["#4C72B0", "#DD8452"], legend=False, showfliers=False, ax=ax)
        ax.set_title(f"Distribución de '{col}' según clase (sin extremos)")
        ax.set_xlabel("Suscripción (y)")
        ax.set_ylabel(col)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_distribucion_numericas.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRÁFICA 4: Matriz de correlación de variables numéricas
    # ---------------------------------------------------------
    df_corr = df.copy()
    df_corr["y_num"] = (df_corr[TARGET] == "yes").astype(int)
    num_all = df_corr.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_all.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de Correlación de Variables Numéricas")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_matriz_correlacion.png", dpi=300)
    plt.close()

    print(f"\nGráficas guardadas correctamente en: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()