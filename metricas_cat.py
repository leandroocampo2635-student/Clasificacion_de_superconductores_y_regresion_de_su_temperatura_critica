import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
df = pd.read_csv("optimized_supercon.csv")

X = df.drop(columns=["Temp_critical"])
y_reg = df["Temp_critical"]
y_cls = (y_reg > 0).astype(int)


# ==========================================
# 2. FUNCIONES DE EVALUACIÓN Y GUARDADO DE PLOTS
# ==========================================
def evaluate_classification_models(
    models_dict,
    X_val,
    y_true_cls,
    csv_out="cat_classification_metrics.csv",
    plot_out="cat_confusion_matrix.png",
):
    """Evalúa modelos de clasificación mediante predict() y predict_proba().

    Genera y guarda la matriz de confusión y exporta métricas a CSV.
    """
    results = []
    n_models = len(models_dict)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models_dict.items()):
        # Inferencia sin reentrenar
        y_pred = model.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)
            y_prob_pos = (
                y_proba[:, 1]
                if y_proba.ndim == 2 and y_proba.shape[1] == 2
                else y_proba
            )
        else:
            y_prob_pos = y_pred

        # Cálculo de métricas
        acc = accuracy_score(y_true_cls, y_pred)
        prec = precision_score(y_true_cls, y_pred, zero_division=0)
        rec = recall_score(y_true_cls, y_pred, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true_cls, y_prob_pos)

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

        # Gráfico Matriz de Confusión
        cm = confusion_matrix(y_true_cls, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            annot_kws={"size": 12},
        )
        ax.set_title(f"Matriz de Confusión: {model_name}", fontsize=12)
        ax.set_xlabel("Clase Predicha (0: No Supercon, 1: Supercon)")
        ax.set_ylabel("Clase Real")

    plt.tight_layout()
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close()

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_out, index=False)
    print(f"-> Métricas de Clasificación guardadas en: {csv_out}")
    print(f"-> Gráfica de Matriz de Confusión guardada en: {plot_out}")
    return df_results


def evaluate_regression_models(
    models_dict,
    X_val,
    y_true_reg,
    csv_out="cat_regression_metrics.csv",
    plot_out="cat_regression_predictions.png",
):
    """Evalúa modelos de regresión usando predict() sobre muestras con Tc > 0.

    Guarda métricas en CSV y gráfica de dispersión en PNG.
    """
    # Filtrar únicamente superconductores para validar el regresor
    mask = y_true_reg > 0
    X_filtered = X_val[mask]
    y_filtered = y_true_reg[mask]

    results = []
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, model in models_dict.items():
        # Predicción directa sobre la escala real
        y_pred = model.predict(X_filtered)

        mae = mean_absolute_error(y_filtered, y_pred)
        rmse = np.sqrt(mean_squared_error(y_filtered, y_pred))
        r2 = r2_score(y_filtered, y_pred)

        results.append(
            {"Modelo": model_name, "MAE": mae, "RMSE": rmse, "R2_Score": r2}
        )

        # Gráfica de Real vs Predicho
        ax.scatter(y_filtered, y_pred, alpha=0.3, label=f"{model_name} (R²={r2:.3f})")

    # Línea ideal 1:1
    max_val = max(y_filtered.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", label="Ideal (1:1)")
    ax.set_xlabel("Temperatura Crítica Real (K)")
    ax.set_ylabel("Temperatura Crítica Predicha (K)")
    ax.set_title("Evaluación Regresor CatBoost (Tc > 0)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close()

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_out, index=False)
    print(f"-> Métricas de Regresión guardadas en: {csv_out}")
    print(f"-> Gráfica de Regresión guardada en: {plot_out}")
    return df_results


# ==========================================
# 3. CARGA DE MODELOS GUARDADOS (SIN FIT)
# ==========================================
print("Cargando modelos entrenados...")
cat_cls = cb.CatBoostClassifier()
cat_cls.load_model("catboost_classifier.cbm")

cat_reg = cb.CatBoostRegressor()
cat_reg.load_model("catboost_regressor.cbm")

cls_dict = {"CatBoost Classifier": cat_cls}
reg_dict = {"CatBoost Regressor": cat_reg}

# ==========================================
# 4. EJECUCIÓN DE VALIDACIÓN Y GENERACIÓN
# ==========================================
print("\n=== EVALUANDO CLASIFICADOR CATBOOST ===")
df_cls = evaluate_classification_models(
    cls_dict,
    X,
    y_cls,
    csv_out="cat_classification_metrics.csv",
    plot_out="cat_confusion_matrix.png",
)
print(df_cls.to_string(index=False))

print("\n=== EVALUANDO REGRESOR CATBOOST (Tc > 0) ===")
df_reg = evaluate_regression_models(
    reg_dict,
    X,
    y_reg,
    csv_out="cat_regression_metrics.csv",
    plot_out="cat_regression_predictions.png",
)
print(df_reg.to_string(index=False))