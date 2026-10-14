import os
import json
import joblib
import warnings
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Optimización para CPU: 4 núcleos en paralelo con 2 hilos (OMP_NUM_THREADS)
# Limitamos los hilos por proceso para evitar el overhead en tu i5-1135G7
os.environ["OMP_NUM_THREADS"] = "2" 

# Suprimir warnings
warnings.filterwarnings("ignore")

# 1. Cargar el dataset optimizado
print("Cargando dataset 'optimized_supercon.csv'...")
df = pd.read_csv("optimized_supercon.csv")

# Asumimos que los superconductores tienen Temp_critical > 0
# Creamos la variable objetivo para clasificación
df['is_supercon'] = (df['Temp_critical'] > 0).astype(int)

# Separación general
X_all = df.drop(columns=['Temp_critical', 'is_supercon'])
y_class = df['is_supercon']
y_reg_all = df['Temp_critical']

# Configuración general de validación cruzada
cv_class = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

def objective_rf_class(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': 4 # Usamos 4 workers (núcleos) en paralelo
    }

    f1_scores = []
    for train_idx, valid_idx in cv_class.split(X_all, y_class):
        X_train, X_valid = X_all.iloc[train_idx], X_all.iloc[valid_idx]
        y_train, y_valid = y_class.iloc[train_idx], y_class.iloc[valid_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_valid)
        f1_scores.append(f1_score(y_valid, preds))

    return np.mean(f1_scores)

def objective_rf_reg(trial):
    # Para la regresión solo usamos los datos que SI son superconductores reales
    mask_supercon = y_class == 1
    X_reg = X_all[mask_supercon]
    y_reg = y_reg_all[mask_supercon]
    
    # Transformación logarítmica debido al sesgo de la temperatura crítica
    y_log = np.log1p(y_reg)
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': 4 # Usamos 4 workers (núcleos) en paralelo
    }

    rmses = []
    # Usar reset_index para evitar problemas de alineación de índices al hacer split
    X_reg = X_reg.reset_index(drop=True)
    y_log = y_log.reset_index(drop=True)

    for train_idx, valid_idx in cv_reg.split(X_reg):
        X_train, X_valid = X_reg.iloc[train_idx], X_reg.iloc[valid_idx]
        y_train, y_valid = y_log.iloc[train_idx], y_log.iloc[valid_idx]

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        preds_log = model.predict(X_valid)
        preds = np.expm1(preds_log)
        y_valid_real = np.expm1(y_valid)

        fold_rmse = np.sqrt(mean_squared_error(y_valid_real, preds))
        rmses.append(fold_rmse)

    return np.mean(rmses)

if __name__ == "__main__":
    n_trials = 100
    sampler = TPESampler(seed=42)

    # ==========================================
    # 1. OPTIMIZACIÓN CLASIFICACIÓN
    # ==========================================
    print("\n" + "="*50)
    print("INICIANDO OPTIMIZACIÓN: CLASIFICACIÓN (¿Es Superconductor?)")
    print("="*50)
    study_class = optuna.create_study(direction="maximize", sampler=sampler, study_name="RF_Class_Supercon")
    study_class.optimize(objective_rf_class, n_trials=n_trials)
    
    # Entrenar el mejor clasificador con TODOS los datos
    best_params_class = study_class.best_params
    best_params_class['n_jobs'] = 4
    best_params_class['random_state'] = 42
    
    print("\nEntrenando clasificador final...")
    final_classifier = RandomForestClassifier(**best_params_class)
    final_classifier.fit(X_all, y_class)

    # ==========================================
    # 2. OPTIMIZACIÓN REGRESIÓN
    # ==========================================
    print("\n" + "="*50)
    print("INICIANDO OPTIMIZACIÓN: REGRESIÓN (Predicción de Temp_critical)")
    print("="*50)
    study_reg = optuna.create_study(direction="minimize", sampler=sampler, study_name="RF_Reg_Supercon")
    study_reg.optimize(objective_rf_reg, n_trials=n_trials)

    # Entrenar el mejor regresor con los datos de superconductores
    best_params_reg = study_reg.best_params
    best_params_reg['n_jobs'] = 4
    best_params_reg['random_state'] = 42
    
    mask_supercon = y_class == 1
    X_reg = X_all[mask_supercon]
    y_reg = y_reg_all[mask_supercon]
    y_reg_log = np.log1p(y_reg)

    print("\nEntrenando regresor final...")
    final_regressor = RandomForestRegressor(**best_params_reg)
    final_regressor.fit(X_reg, y_reg_log)

    # ==========================================
    # 3. GUARDADO DE MODELOS Y PARÁMETROS
    # ==========================================
    print("\nGuardando modelos y parámetros...")
    
    # Guardar modelos usando joblib (ideal para scikit-learn y SHAP posterior)
    joblib.dump(final_classifier, "rf_classifier_model.pkl")
    joblib.dump(final_regressor, "rf_regressor_model.pkl")
    
    # Guardar parámetros en JSON
    hiperparametros = {
        "clasificacion_best_score_f1": study_class.best_value,
        "clasificacion_params": best_params_class,
        "regresion_best_rmse": study_reg.best_value,
        "regresion_params": best_params_reg
    }
    
    with open("rf_hyperparameters.json", "w") as f:
        json.dump(hiperparametros, f, indent=4)

    # ==========================================
    # 4. RESULTADOS FINALES
    # ==========================================
    print("\n" + "="*40)
    print("RESUMEN DE RESULTADOS")
    print("="*40)
    print(f"Mejor F1-Score (Clasificación): {study_class.best_value:.4f}")
    print(f"Mejor RMSE (Regresión): {study_reg.best_value:.4f} K")
    print("\nLos modelos han sido guardados como .pkl para uso futuro con SHAP.")
    print("Los parámetros se han guardado en rf_hyperparameters.json")