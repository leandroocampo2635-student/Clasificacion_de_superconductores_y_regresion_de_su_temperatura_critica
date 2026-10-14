import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# -------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# -------------------------------------------------------------------------
df = pd.read_csv("optimized_supercon.csv")

X = df.drop(columns=["Temp_critical"])
y_reg = df["Temp_critical"]
y_cls = (y_reg > 0).astype(int)

y_log = np.log1p(y_reg)
# -------------------------------------------------------------------------
# 2. FUNCIONES DE EVALUACIÓN SIN REENTRENAR (SOLO .predict() Y .predict_proba())
# -------------------------------------------------------------------------
def evaluate_classifiers(
    models_dict, X, y_true, csv_out="classification_metrics.csv"
):
    """Evalúa un diccionario de clasificadores entrenados, genera la matriz de confusión

    y guarda las métricas comparativas en un CSV.
    """
    results = []
    n_models = len(models_dict)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models_dict.items()):
        # Predicción de clases y probabilidades
        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            # Extraer probabilidad de la clase positiva (1)
            y_prob_pos = (
                y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
            )
        else:
            y_prob_pos = y_pred

        # Cálculo de métricas
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob_pos)

        results.append(
            {
                "Modelo": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
            }
        )

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Matriz de Confusión: {model_name}")
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    plt.tight_layout()
    plt.show()

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_out, index=False)
    print(f"Métricas de clasificación guardadas en: {csv_out}")
    return df_results


def evaluate_regressors(
    models_dict, X, y_true, csv_out="regression_metrics.csv"
):
    """Evalúa un diccionario de regresores entrenados y guarda las métricas

    comparativas en un CSV.
    """
    results = []

    for model_name, model in models_dict.items():
        # Únicamente predicción de valores continuos
        y_pred = model.predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        results.append(
            {"Modelo": model_name, "MAE": mae, "RMSE": rmse, "R2_Score": r2}
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_out, index=False)
    print(f"Métricas de regresión guardadas en: {csv_out}")
    return df_results


# -------------------------------------------------------------------------
# 3. CARGA DE MODELOS PREVIAMENTE ENTRENADOS Y SUS HIPERPARÁMETROS JSON
# -------------------------------------------------------------------------

# --- Cargar Hiperparámetros (Opcional para inspección/logs) ---
with open("best_hyperparameters.json", "r") as f:
    lgbm_params = json.load(f)

with open("best_params_classifier.json", "r") as f:
    catboost_cls_params = json.load(f)

with open("best_params_regressor.json", "r") as f:
    catboost_reg_params = json.load(f)

with open("rf_hyperparameters.json", "r") as f:
    rf_params = json.load(f)

# --- Cargar Modelos de Clasificación ---
lgbm_cls = joblib.load("lgbm_classifier.pkl")

catboost_cls = CatBoostClassifier()
catboost_cls.load_model("catboost_classifier.cbm")

rf_cls = joblib.load("rf_classifier_model.pkl")

# --- Cargar Modelos de Regresión ---
lgbm_reg = joblib.load("lgbm_regressor.pkl")

catboost_reg = CatBoostRegressor()
catboost_reg.load_model("catboost_regressor.cbm")

rf_reg = joblib.load("rf_regressor_model.pkl")


# -------------------------------------------------------------------------
# 4. EJECUCIÓN Y GENERACIÓN DE RESULTADOS
# -------------------------------------------------------------------------
classification_models = {
    "LightGBM": lgbm_cls,
    "CatBoost": catboost_cls,
    "Random Forest": rf_cls,
}

regression_models = {
    "LightGBM": lgbm_reg,
    "CatBoost": catboost_reg,
    "Random Forest": rf_reg,
}

# Evaluar clasificadores (genera gráficos y 'classification_metrics.csv')
df_cls_metrics = evaluate_classifiers(
    classification_models,
    X,
    y_cls,
    csv_out="classification_results_comparison.csv",
)

# Evaluar regresores (genera 'regression_metrics.csv')
df_reg_metrics = evaluate_regressors(
    regression_models, X, y_reg, csv_out="regression_results_comparison.csv"
)

# Mostrar resultados en consola
print("\n=== COMPARATIVA DE CLASIFICACIÓN ===")
print(df_cls_metrics.to_string(index=False))

print("\n=== COMPARATIVA DE REGRESIÓN ===")
print(df_reg_metrics.to_string(index=False))