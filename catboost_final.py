import json
import warnings
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")


# 1. CARGA Y PREPARACIÓN DE DATOS
df = pd.read_csv("optimized_supercon.csv")

X = df.drop(columns=["Temp_critical"])
y_reg = df["Temp_critical"]  # Escala original en Kelvin 
y_cls = (y_reg > 0).astype(int)

# Subconjunto exclusivo de superconductores para entrenar la regresión (Tc > 0)
mask_supercon = y_cls == 1
X_supercon = X[mask_supercon].reset_index(drop=True)
y_supercon = y_reg[mask_supercon].reset_index(drop=True)



# 2. Entrenamiento del modelo secuencial

def predict_superconductor_pipeline(X_input, classifier_model, regressor_model):
    """Inferencia secuencial en dos etapas:

    1. El clasificador evalúa si la muestra es superconductora (1) o no (0).
    2. Si el clasificador predice 1, pasa al regresor para estimar Temp_critical.
    3. Si el clasificador predice 0, asigna automáticamente 0.0 K.
    """
    # Predicción binaria de clasificación
    preds_cls = classifier_model.predict(X_input)

    # Arreglo base inicializado en 0.0 K
    preds_reg = np.zeros(len(X_input))

    # Filtrar índices identificados como superconductores (clase 1)
    supercon_indices = np.where(preds_cls == 1)[0]

    # Pasar al regresor solo los verdaderos/falsos positivos del clasificador
    if len(supercon_indices) > 0:
        if isinstance(X_input, pd.DataFrame):
            X_to_regress = X_input.iloc[supercon_indices]
        else:
            X_to_regress = X_input[supercon_indices]

        preds_reg[supercon_indices] = regressor_model.predict(X_to_regress)

    return preds_cls, preds_reg



# 3. OPTIMIZACIÓN CLASIFICADOR (CatBoost)

def objective_classifier(trial):
    params = {
        "thread_count": 2,
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.1, log=True
        ),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "iterations": 1000,
        "random_seed": 42,
        "verbose": False,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, valid_idx in skf.split(X, y_cls):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_cls.iloc[train_idx], y_cls.iloc[valid_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=30,
            verbose=False,
        )
        preds_prob = model.predict_proba(X_valid)[:, 1]

        aucs.append(roc_auc_score(y_valid, preds_prob))

    return np.mean(aucs)



# 4. OPTIMIZACIÓN REGRESOR (CatBoost)
def objective_regressor(trial):
    params = {
        "thread_count": 2,
        "depth": trial.suggest_int("depth", 4, 9),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.1, log=True
        ),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
        "iterations": 1500,
        "random_seed": 42,
        "verbose": False,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, valid_idx in kf.split(X_supercon):
        X_train, X_valid = X_supercon.iloc[train_idx], X_supercon.iloc[valid_idx]
        y_train, y_valid = y_supercon.iloc[train_idx], y_supercon.iloc[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=40,
            verbose=False,
        )
        preds = model.predict(X_valid)

        rmses.append(np.sqrt(mean_squared_error(y_valid, preds)))

    return np.mean(rmses)



# 5. EJECUCIÓN, GUARDADO Y PRUEBA EN CASCADA

if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    n_trials = 100

    # CLASIFICADOR
    if y_cls.nunique() > 1:
        print("1/3 Optimizando Clasificador CatBoost...")
        study_cls = optuna.create_study(direction="maximize", sampler=sampler)
        study_cls.optimize(objective_classifier, n_trials=n_trials)

        best_params_cls = study_cls.best_params
        best_params_cls.update(
            {"thread_count": 2, "random_seed": 42, "iterations": 1000, "verbose": False}
        )

        final_cls_model = CatBoostClassifier(**best_params_cls)
        final_cls_model.fit(X, y_cls, verbose=False)

        final_cls_model.save_model("catboost_classifier.cbm")
        with open("catboost_best_params_classifier.json", "w") as f:
            json.dump(best_params_cls, f, indent=4)

        print(f"  -> ROC-AUC Clasificación: {study_cls.best_value:.4f}")
    else:
        print(
            "Dataset contiene una sola clase. Omitiendo optimización de clasificación."
        )

    # REGRESOR 
    print("\n2/3 Optimizando Regresor (CatBoost - Escala original K)...")
    study_reg = optuna.create_study(direction="minimize", sampler=sampler)
    study_reg.optimize(objective_regressor, n_trials=n_trials)

    best_params_reg = study_reg.best_params
    best_params_reg.update(
        {"thread_count": 2, "random_seed": 42, "iterations": 1500, "verbose": False}
    )

    final_reg_model = CatBoostRegressor(**best_params_reg)
    final_reg_model.fit(X_supercon, y_supercon, verbose=False)

    final_reg_model.save_model("catboost_regressor.cbm")
    with open("catboost_best_params_regressor.json", "w") as f:
        json.dump(best_params_reg, f, indent=4)

    print(f"  -> RMSE Regresión: {study_reg.best_value:.4f} K")

    # --- PRUEBA DEL PIPELINE EN CASCADA COMPLETO ---
    print("\n3/3 Evaluando el flujo secuencial (Clasificador -> Regresor)...")
    y_pred_cls, y_pred_reg = predict_superconductor_pipeline(
        X, final_cls_model, final_reg_model
    )

    # Cálculo de métricas integradas
    acc = accuracy_score(y_cls, y_pred_cls)
    f1 = f1_score(y_cls, y_pred_cls)
    mae_global = mean_absolute_error(y_reg, y_pred_reg)
    rmse_global = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
    r2_global = r2_score(y_reg, y_pred_reg)

    print("\n=== RESULTADOS GLOBALES DEL PIPELINE COMPLETO ===")
    print(f"Accuracy Clasificación : {acc:.4f}")
    print(f"F1-Score Clasificación : {f1:.4f}")
    print(f"MAE Global Regresión    : {mae_global:.4f} K")
    print(f"RMSE Global Regresión   : {rmse_global:.4f} K")
    print(f"R2 Global Regresión     : {r2_global:.4f}")

    print("\n¡Proceso completado! Archivos y modelos guardados exitosamente.")