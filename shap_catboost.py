import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from catboost import CatBoostClassifier, CatBoostRegressor

# Suprimir advertencias menores para consola limpia
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# OPTIMIZACIÓN DE HARDWARE: Intel Core i5-1135G7
# 4 Núcleos físicos / 8 Hilos de procesamiento
# ---------------------------------------------------------
# Estas variables de entorno aceleran las operaciones subyacentes 
# de C++/NumPy/SciPy utilizadas internamente por SHAP.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

# Desactivar interactividad de pyplot para guardar gráficos rápidamente
# y liberar la carga en la memoria RAM.
plt.ioff()
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("=" * 65)
print("INICIANDO ANÁLISIS DE EXPLICABILIDAD SHAP PARA CATBOOST")
print("Hardware objetivo: Intel Core i5-1135G7 (8 hilos activados)")
print("=" * 65)

# 1. Cargar el Dataset
dataset_path = "optimized_supercon.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"No se encontró el archivo '{dataset_path}'.")

print("Cargando dataset...")
df = pd.read_csv(dataset_path)

# Limpiar columnas irrelevantes si existen (similar a la fase de entrenamiento)
columns_to_drop = [col for col in ['Material-ID', 'Chemical_Formula'] if col in df.columns]
if columns_to_drop:
    df = df.drop(columns=columns_to_drop)

X = df.drop(columns=['Temp_critical'])
y_reg = df['Temp_critical']
y_cls = (y_reg > 0).astype(int)

# 2. Cargar Hiperparámetros y Modelos
json_path = "best_hyperparameters.json"
clf_path = "catboost_classifier.cbm"
reg_path = "catboost_regressor.cbm"

# Cargar parámetros para registro/contexto (opcional para SHAP, 
# pero útil para verificar la consistencia).
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        hyperparams = json.load(f)
    print(f"Hiperparámetros base cargados (Trials: {hyperparams.get('n_trials_per_stage', 'N/A')})")
else:
    print(f"Advertencia: No se encontró '{json_path}'. Se continuará sin él.")

# Cargar Modelos Entrenados (.cbm)
clf_model = CatBoostClassifier()
if os.path.exists(clf_path):
    clf_model.load_model(clf_path)
    print("Modelo clasificador CatBoost cargado.")
else:
    clf_model = None

reg_model = CatBoostRegressor()
if os.path.exists(reg_path):
    reg_model.load_model(reg_path)
    print("Modelo regresor CatBoost cargado.")
else:
    reg_model = None

# Crear directorio de salidas para las gráficas
output_dir = "shap_results_catboost"
os.makedirs(output_dir, exist_ok=True)
print(f"Las gráficas se guardarán en la carpeta: {output_dir}/")


# 3. SHAP PARA EL CLASIFICADOR 

if clf_model is not None and y_cls.nunique() > 1:
    print("\n[1/2] Calculando valores SHAP para el CLASIFICADOR...")
    
    # TreeExplainer de SHAP es el método exacto y rápido para modelos basados en árboles
    explainer_clf = shap.TreeExplainer(clf_model)
    
    # OPTIMIZACIÓN CPU: Para evitar tiempos de cálculo excesivos
    # en el i5-1135G7, usamos un muestreo representativo de 5000 instancias.
    X_sample_clf = X if len(X) <= 5000 else X.sample(n=5000, random_state=42)
    
    # Calculando valores
    shap_values_clf = explainer_clf(X_sample_clf)

    # 3.1 Gráfico de Resumen (Beeswarm Plot) - Clasificador
    plt.figure(figsize=(10, 6))
    # Para CatBoost binario, SHAP a veces devuelve un tensor. Aseguramos el formato adecuado:
    shap.plots.beeswarm(shap_values_clf, max_display=15, show=False)
    plt.title("SHAP Beeswarm - Clasificador (Superconductor vs No)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_clf_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3.2 Gráfico de Importancia Global (Bar Plot) - Clasificador
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_clf, max_display=15, show=False)
    plt.title("Importancia Global de Características - Clasificador", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_clf_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(" -> Gráficos del clasificador generados y guardados con éxito.")
else:
    print("\n[1/2] Omitiendo clasificador (modelo no encontrado o dataset inválido).")

# ==============================================================================
# 4. SHAP PARA EL REGRESOR (Predicción de Temperatura Crítica)
# ==============================================================================
if reg_model is not None:
    print("\n[2/2] Calculando valores SHAP para el REGRESOR...")
    
    # Filtramos solo aquellos que SÍ son superconductores para analizar la regresión
    mask_supercon = y_cls == 1
    X_supercon = X[mask_supercon].reset_index(drop=True)
    
    explainer_reg = shap.TreeExplainer(reg_model)
    
    # OPTIMIZACIÓN CPU: Muestreo a 5000 para el regresor
    X_sample_reg = X_supercon if len(X_supercon) <= 5000 else X_supercon.sample(n=5000, random_state=42)
    
    # Calculando valores
    shap_values_reg = explainer_reg(X_sample_reg)

    # 4.1 Gráfico de Resumen (Beeswarm Plot) - Regresor
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_reg, max_display=15, show=False)
    plt.title("SHAP Beeswarm - Impacto en la Temp. Crítica (Regresor)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_reg_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4.2 Gráfico de Importancia Global (Bar Plot) - Regresor
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_reg, max_display=15, show=False)
    plt.title("Importancia Global de Características - Regresor", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_reg_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.3 Gráfico de Dependencia (Scatter)
    # Busca la característica con el mayor valor SHAP promedio absoluto
    top_feature_idx = np.argsort(-np.abs(shap_values_reg.values).mean(0))[0]
    top_feature_name = X_sample_reg.columns[top_feature_idx]
    
    plt.figure(figsize=(8, 6))
    shap.plots.scatter(shap_values_reg[:, top_feature_name], color=shap_values_reg, show=False)
    plt.title(f"Dependencia SHAP: {top_feature_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_reg_dependence.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(" -> Gráficos del regresor generados y guardados con éxito.")
else:
    print("\n[2/2] Omitiendo regresor (modelo 'catboost_regressor.cbm' no encontrado).")

print("\n" + "="*65)
print("¡ANÁLISIS SHAP FINALIZADO!")
print(f"Revisa la carpeta '{output_dir}' para ver tus gráficas de explicabilidad.")
print("="*65)