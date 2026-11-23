Clasificación Supervisada en *Bank Marketing* sin Fuga de Información Temporal

Proyecto final de Ciencia de Datos enfocado en la predicción de suscripción a depósitos a plazo fijo sobre el conjunto de datos **Bank Marketing** (UCI Machine Learning Repository), evaluando un escenario realista previo a la llamada telefónica (*pre-call scoring*).

1. Estructura del Proyecto

```text
Proyecto_Data_Science/
├── data/
│   ├── raw/                  # Dataset original (bank-full.csv)
│   └── processed/            # Particiones estratificadas limpias (train/val/test)
├── models/                   # Modelos entrenados y preprocesadores (.joblib, .keras)
├── reports/
│   ├── figures/              # Gráficas de EDA, matrices de confusión y curvas ROC/PR (300 DPI)
│   ├── analisis_errores_test.csv
│   ├── comparativa_modelos.csv
│   └── espacios_busqueda.json
├── src/
│   ├── config.py             # Rutas, hiperparámetros globales y semilla fija (RANDOM_STATE = 42)
│   ├── eda.py                # Análisis exploratorio de datos y generación de figuras 01-04
│   ├── preprocess.py         # Partición estratificada 70/15/15 y preprocesamiento sin fuga
│   └── train.py              # Entrenamiento, validación cruzada, métricas y análisis de errores
├── README.md
└── requirements.txt

2. Requisitos e Instalación
El proyecto fue desarrollado en Python 3.14 con semilla fija (RANDOM_STATE = 42) para garantizar la reproducibilidad de las particiones y resultados.

Crear y activar el entorno virtual:

Bash
python -m venv .banco
# En Windows (PowerShell):
.\.banco\Scripts\Activate.ps1
# En Linux/macOS:
source .banco/bin/activate

Instalar las dependencias exactas:

Bash
pip install -r requirements.txt


3. Instrucciones de Ejecución
Ejecutar los módulos desde la raíz del proyecto en el siguiente orden:

Análisis Exploratorio de Datos (EDA): Genera las gráficas exploratorias (01 a 04) en reports/figures/.

Bash
python -m src.eda
Preprocesamiento y Partición Estratificada: Elimina la variable duration para evitar fuga temporal (target leakage), divide en 70% entrenamiento, 15% validación y 15% prueba, y ajusta las transformaciones únicamente sobre entrenamiento.

Bash
python -m src.preprocess
Entrenamiento, Ajuste y Evaluación: Ejecuta búsqueda de hiperparámetros con Pipeline y validación cruzada estratificada (Regresión Logística, Gaussian Naive Bayes, SVM lineal/RBF y Perceptrón Multicapa en Keras), exportando modelos, métricas, curvas ROC/Precision-Recall (05 a 10) y el reporte de errores.

Bash
python -m src.train

## 4. Resumen de Resultados en Prueba (N = 6,782)

| Modelo | Hiperparámetros Óptimos | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Regresión Logística (Base) | `C=10.0`, `lbfgs` | 0.7630 | 0.2669 | 0.5876 | 0.3671 | 0.7386 | 0.3668 |
| Gaussian Naive Bayes | `var_smoothing=1e-11` | 0.8453 | 0.3581 | 0.4073 | 0.3811 | 0.7237 | 0.3272 |
| SVM (RBF) | `C=5.0`, `gamma=0.01` | 0.8440 | 0.3765 | 0.5095 | 0.4330 | 0.7620 | 0.3954 |
| Red Neuronal (MLP) | `64-32`, `dropout=0.2`, `lr=5e-4` | 0.8309 | 0.3573 | 0.5586 | **0.4358** | **0.7802** | **0.4124** |