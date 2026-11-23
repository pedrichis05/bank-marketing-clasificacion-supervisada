import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from preprocess import build_preprocessor, get_raw_splits, load_bank_data


def build_neural_network(
    input_dim: int,
    units_1: int = 64,
    units_2: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> Sequential:
    """Arquitectura secuencial reciclada de clase, parametrizada para búsqueda."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units_1, activation="relu"),
        Dropout(dropout_rate),
        Dense(units_2, activation="relu"),
        Dropout(dropout_rate / 2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model_test(
    nombre: str,
    y_true,
    y_pred,
    y_prob,
    fig_prefix: str,
    val_f1_mean: float,
    val_f1_std: float,
    val_auc_mean: float,
    val_auc_std: float,
    criterio_var: str,
    mejores_params: str,
) -> dict:
    """Calcula métricas en prueba y exporta matriz de confusión a 300 DPI."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    print(f"\n--- Resultados en Prueba: {nombre} ---")
    print(f"Hiperparámetros        : {mejores_params}")
    print(f"Val F1 ({criterio_var})  : {val_f1_mean:.4f} ± {val_f1_std:.4f}")
    print(f"Val AUC ({criterio_var}) : {val_auc_mean:.4f} ± {val_auc_std:.4f}")
    print(f"Test Accuracy          : {acc:.4f}")
    print(f"Test Precision         : {prec:.4f}")
    print(f"Test Recall            : {rec:.4f}")
    print(f"Test F1-Score          : {f1:.4f}")
    print(f"Test ROC-AUC           : {roc_auc:.4f}")
    print(f"Test PR-AUC            : {pr_auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.grid(False)
    ax.set_title(f"Matriz de Confusión - {nombre}")
    plt.tight_layout()
    slug = nombre.lower().replace(" ", "_").replace("ó", "o").replace("í", "i")
    plt.savefig(FIGURES_DIR / f"{fig_prefix}_cm_{slug}.png", dpi=300)
    plt.close()

    return {
        "Modelo": nombre,
        "Criterio_Variabilidad": criterio_var,
        "Mejores_Hiperparametros": mejores_params,
        "Val_F1_Media": round(val_f1_mean, 4),
        "Val_F1_Std": round(val_f1_std, 4),
        "Val_AUC_Media": round(val_auc_mean, 4),
        "Val_AUC_Std": round(val_auc_std, 4),
        "Test_Accuracy": round(acc, 4),
        "Test_Precision": round(prec, 4),
        "Test_Recall": round(rec, 4),
        "Test_F1": round(f1, 4),
        "Test_ROC_AUC": round(roc_auc, 4),
        "Test_PR_AUC": round(pr_auc, 4),
    }


def extract_grid_cv_stats(grid: GridSearchCV) -> tuple[float, float, float, float]:
    """Extrae media y std exactas de F1 y ROC-AUC directamente de GridSearchCV sin re-evaluar."""
    idx = grid.best_index_
    res = grid.cv_results_
    return (
        float(res["mean_test_f1"][idx]),
        float(res["std_test_f1"][idx]),
        float(res["mean_test_roc_auc"][idx]),
        float(res["std_test_roc_auc"][idx]),
    )


def plot_roc_and_pr_curves(curves_data: dict) -> None:
    """Genera curvas ROC y Precision-Recall comparativas en el conjunto de prueba."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colores = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    for (nombre, (y_true, y_prob)), color in zip(curves_data.items(), colores):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{nombre} (AUC = {auc_val:.4f})", color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio (0.5000)")
    axes[0].set_title("Curvas ROC Comparativas (Conjunto de Prueba)")
    axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Sensibilidad / Recall (TPR)")
    axes[0].legend(loc="lower right", fontsize=8.5)

    baseline_pr = np.mean(next(iter(curves_data.values()))[0])
    for (nombre, (y_true, y_prob)), color in zip(curves_data.items(), colores):
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        axes[1].plot(rec_vals, prec_vals, label=f"{nombre} (AP = {ap_val:.4f})", color=color, linewidth=2)

    axes[1].axhline(baseline_pr, linestyle="--", color="gray", label=f"Línea base ({baseline_pr:.4f})")
    axes[1].set_title("Curvas Precision-Recall (Clase Minoritaria)")
    axes[1].set_xlabel("Sensibilidad / Recall")
    axes[1].set_ylabel("Precisión (Precision)")
    axes[1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_curvas_roc_pr_comparativas.png", dpi=300)
    plt.close()


def plot_nn_history(history) -> None:
    """Gráficas de Accuracy y Loss para documentar el entrenamiento de la Red Neuronal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="Entrenamiento", color="#4C72B0")
    axes[0].plot(history.history["val_accuracy"], label="Validación", color="#DD8452")
    axes[0].set_title("Evolución de Exactitud (Red Neuronal)")
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Entrenamiento", color="#4C72B0")
    axes[1].plot(history.history["val_loss"], label="Validación", color="#DD8452")
    axes[1].set_title("Evolución de Pérdida (Binary Crossentropy)")
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_curvas_red_neuronal.png", dpi=300)
    plt.close()


def export_error_analysis(y_test, preds_dict: dict, probs_dict: dict) -> None:
    """Exporta casos de prueba con variables originales (incluyendo duration solo como referencia)."""
    df_raw = load_bank_data()
    X_raw_full = df_raw.drop(columns=[TARGET])
    y_raw = (df_raw[TARGET] == "yes").astype(int)

    _, X_test_ref, _, y_test_check = train_test_split(
        X_raw_full, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_raw
    )
    assert np.array_equal(y_test.values, y_test_check.values), "Desalineación en partición de test"

    df_errors = X_test_ref.copy().reset_index(drop=True)
    df_errors["y_real"] = y_test.values

    for key in preds_dict:
        df_errors[f"pred_{key}"] = preds_dict[key]
        df_errors[f"prob_{key}"] = np.round(probs_dict[key], 4)

    df_errors.to_csv(REPORTS_DIR / "analisis_errores_test.csv", index=False)


def run_training() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # Datos crudos sin 'duration' para evitar fuga entre folds y antes de la llamada
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = get_raw_splits()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    resultados = []
    curves_data = {}
    preds_dict = {}
    probs_dict = {}
    search_spaces_log = {}

    # =========================================================
    # 1. MODELO DE REFERENCIA: Regresión Logística (Pipeline CV)
    # =========================================================
    print("\n[1/4] Ajustando Regresión Logística (Pipeline 5-fold CV)...")
    lr_pipe = Pipeline([
        ("prep", build_preprocessor(X_train_raw)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    lr_param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
    }
    search_spaces_log["Regresion_Logistica"] = lr_param_grid

    lr_grid = GridSearchCV(
        lr_pipe,
        param_grid=lr_param_grid,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1",
        cv=cv,
        n_jobs=-1,
    )
    lr_grid.fit(X_train_raw, y_train)
    best_lr = lr_grid.best_estimator_
    joblib.dump(best_lr, MODELS_DIR / "logistic_regression_pipeline.joblib")

    f1_m, f1_s, auc_m, auc_s = extract_grid_cv_stats(lr_grid)
    y_pred_lr = best_lr.predict(X_test_raw)
    y_prob_lr = best_lr.predict_proba(X_test_raw)[:, 1]

    resultados.append(evaluate_model_test(
        "Regresión Logística", y_test, y_pred_lr, y_prob_lr, "05",
        f1_m, f1_s, auc_m, auc_s, "5-fold CV", str(lr_grid.best_params_),
    ))
    curves_data["Regresión Logística"] = (y_test, y_prob_lr)
    preds_dict["lr"] = y_pred_lr
    probs_dict["lr"] = y_prob_lr

    # =========================================================
    # 2. CLASIFICADOR BAYESIANO: GaussianNB (Pipeline CV)
    # =========================================================
    print("\n[2/4] Ajustando Gaussian Naive Bayes (Pipeline 5-fold CV)...")
    nb_pipe = Pipeline([
        ("prep", build_preprocessor(X_train_raw)),
        ("clf", GaussianNB()),
    ])
    nb_param_grid = {
        "clf__var_smoothing": np.logspace(-11, -5, 7).tolist(),
    }
    search_spaces_log["Naive_Bayes"] = nb_param_grid

    nb_grid = GridSearchCV(
        nb_pipe,
        param_grid=nb_param_grid,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1",
        cv=cv,
        n_jobs=-1,
    )
    nb_grid.fit(X_train_raw, y_train)
    best_nb = nb_grid.best_estimator_
    joblib.dump(best_nb, MODELS_DIR / "naive_bayes_pipeline.joblib")

    f1_m, f1_s, auc_m, auc_s = extract_grid_cv_stats(nb_grid)
    y_pred_nb = best_nb.predict(X_test_raw)
    y_prob_nb = best_nb.predict_proba(X_test_raw)[:, 1]

    resultados.append(evaluate_model_test(
        "Naive Bayes", y_test, y_pred_nb, y_prob_nb, "06",
        f1_m, f1_s, auc_m, auc_s, "5-fold CV", str(nb_grid.best_params_),
    ))
    curves_data["Naive Bayes"] = (y_test, y_prob_nb)
    preds_dict["nb"] = y_pred_nb
    probs_dict["nb"] = y_prob_nb

    # =========================================================
    # 3. MÁQUINA DE SOPORTE VECTORIAL: SVM Linear y RBF (Pipeline CV)
    # =========================================================
    print("\n[3/4] Ajustando SVM Linear y RBF (Pipeline CV sobre submuestra estratificada)...")
    svm_pipe = Pipeline([
        ("prep", build_preprocessor(X_train_raw)),
        ("clf", SVC(class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    svm_param_grid = [
        {"clf__kernel": ["linear"], "clf__C": [0.1, 1.0, 10.0]},
        {"clf__kernel": ["rbf"], "clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", 0.01, 0.1]},
    ]
    search_spaces_log["SVM"] = svm_param_grid

    X_tune_raw, _, y_tune, _ = train_test_split(
        X_train_raw, y_train, train_size=6000, stratify=y_train, random_state=RANDOM_STATE
    )
    svm_grid = GridSearchCV(
        svm_pipe,
        param_grid=svm_param_grid,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1",
        cv=cv,
        n_jobs=-1,
    )
    svm_grid.fit(X_tune_raw, y_tune)
    f1_m, f1_s, auc_m, auc_s = extract_grid_cv_stats(svm_grid)

    # Reentrenamiento final con TODO X_train_raw y probability=True
    best_svm_params = {k.replace("clf__", ""): v for k, v in svm_grid.best_params_.items()}
    best_svm_pipe = Pipeline([
        ("prep", build_preprocessor(X_train_raw)),
        ("clf", SVC(**best_svm_params, class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
    ])
    best_svm_pipe.fit(X_train_raw, y_train)
    joblib.dump(best_svm_pipe, MODELS_DIR / "svm_best_pipeline.joblib")

    y_pred_svm = best_svm_pipe.predict(X_test_raw)
    y_prob_svm = best_svm_pipe.predict_proba(X_test_raw)[:, 1]

    resultados.append(evaluate_model_test(
        "SVM", y_test, y_pred_svm, y_prob_svm, "07",
        f1_m, f1_s, auc_m, auc_s, "5-fold CV (n=6000)", str(svm_grid.best_params_),
    ))
    curves_data["SVM"] = (y_test, y_prob_svm)
    preds_dict["svm"] = y_pred_svm
    probs_dict["svm"] = y_prob_svm

    # =========================================================
    # 4. RED NEURONAL (Ajustado solo con X_train_raw)
    # =========================================================
    print("\n[4/4] Ajustando Red Neuronal (Keras MLP + EarlyStopping)...")
    nn_preprocessor = build_preprocessor(X_train_raw)
    X_train_nn = nn_preprocessor.fit_transform(X_train_raw)
    X_val_nn = nn_preprocessor.transform(X_val_raw)
    X_test_nn = nn_preprocessor.transform(X_test_raw)
    joblib.dump(nn_preprocessor, MODELS_DIR / "nn_preprocessor.joblib")

    clases = np.unique(y_train)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_train)
    class_weights = dict(zip(clases, pesos))

    nn_configs = [
        {"units_1": 64, "units_2": 32, "dropout_rate": 0.3, "learning_rate": 1e-3},
        {"units_1": 128, "units_2": 64, "dropout_rate": 0.3, "learning_rate": 1e-3},
        {"units_1": 64, "units_2": 32, "dropout_rate": 0.2, "learning_rate": 5e-4},
    ]
    search_spaces_log["Red_Neuronal"] = nn_configs

    mejor_val_f1 = -1.0
    mejor_cfg = None

    for cfg in nn_configs:
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        model_tmp = build_neural_network(X_train_nn.shape[1], **cfg)
        es_tmp = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model_tmp.fit(
            X_train_nn, y_train,
            validation_data=(X_val_nn, y_val),
            epochs=80,
            batch_size=128,
            class_weight=class_weights,
            callbacks=[es_tmp],
            verbose=0,
        )
        val_preds = (model_tmp.predict(X_val_nn, verbose=0).ravel() >= 0.5).astype(int)
        val_f1 = f1_score(y_val, val_preds)
        if val_f1 > mejor_val_f1:
            mejor_val_f1 = val_f1
            mejor_cfg = cfg

    # Variabilidad entre 5 semillas de inicialización aleatoria
    semillas = [42, 100, 2026, 7, 99]
    nn_f1_runs = []
    nn_auc_runs = []
    best_nn = None
    best_history = None

    for idx, seed in enumerate(semillas):
        tf.keras.backend.clear_session()
        np.random.seed(seed)
        tf.random.set_seed(seed)

        model_run = build_neural_network(X_train_nn.shape[1], **mejor_cfg)
        early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
        hist = model_run.fit(
            X_train_nn, y_train,
            validation_data=(X_val_nn, y_val),
            epochs=120,
            batch_size=128,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0,
        )
        val_probs = model_run.predict(X_val_nn, verbose=0).ravel()
        val_preds = (val_probs >= 0.5).astype(int)
        nn_f1_runs.append(f1_score(y_val, val_preds))
        nn_auc_runs.append(roc_auc_score(y_val, val_probs))

        if idx == 0:
            best_nn = model_run
            best_history = hist

    best_nn.save(MODELS_DIR / "red_neuronal.keras")
    plot_nn_history(best_history)

    y_prob_nn = best_nn.predict(X_test_nn, verbose=0).ravel()
    y_pred_nn = (y_prob_nn >= 0.5).astype(int)

    resultados.append(evaluate_model_test(
        "Red Neuronal", y_test, y_pred_nn, y_prob_nn, "08",
        float(np.mean(nn_f1_runs)), float(np.std(nn_f1_runs)),
        float(np.mean(nn_auc_runs)), float(np.std(nn_auc_runs)),
        "5 semillas (Val)", str(mejor_cfg),
    ))
    curves_data["Red Neuronal"] = (y_test, y_prob_nn)
    preds_dict["nn"] = y_pred_nn
    probs_dict["nn"] = y_prob_nn

    # =========================================================
    # 5. EXPORTAR RESULTADOS, CURVAS Y ANÁLISIS DE ERRORES
    # =========================================================
    plot_roc_and_pr_curves(curves_data)
    export_error_analysis(y_test, preds_dict, probs_dict)

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(REPORTS_DIR / "comparativa_modelos.csv", index=False)

    with open(REPORTS_DIR / "espacios_busqueda.json", "w", encoding="utf-8") as f:
        json.dump(search_spaces_log, f, indent=4, ensure_ascii=False)

    print("\n================ TABLA COMPARATIVA FINAL ================")
    print(df_resultados.drop(columns=["Mejores_Hiperparametros"]).to_string(index=False))


if __name__ == "__main__":
    run_training()