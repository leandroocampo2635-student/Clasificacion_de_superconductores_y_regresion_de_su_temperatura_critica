import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings

# Suprimir advertencias menores para mantener la consola limpia
warnings.filterwarnings("ignore")

# ==========================================
# OPTIMIZACIÓN PARA CPU: i5-1135G7
# ==========================================
# Tu procesador tiene 4 núcleos físicos y 8 hilos. 
# Para evitar overhead por cambio de contexto al usar los 4 núcleos en paralelo, 
# limitamos los subprocesos de las librerías matemáticas a 2 hilos.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# Desactivar interactividad de pyplot para guardar gráficos sin consumir RAM extra
plt.ioff()
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("=" * 60)
print("INICIANDO ANÁLISIS DE EXPLICABILIDAD CON SHAP (RANDOM FOREST)")
print("Procesador: Intel Core i5-1135G7 (Optimizado a 2 hilos/proceso)")
print("=" * 60)

# 1. Cargar el Dataset
dataset_path = "optimized_supercon.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"No se encontró el archivo '{dataset_path}'. Asegúrate de estar en el directorio correcto.")

print("Cargando dataset...")
df = pd.read_csv(dataset_path)

# Separar características y variables objetivo
X = df.drop(columns=['Temp_critical', 'is_supercon'], errors='ignore')
y_reg = df['Temp_critical']
y_cls = (y_reg > 0).astype(int)

# 2. Cargar Modelos e Hiperparámetros
clf_path = 'rf_classifier_model.pkl'
reg_path = 'rf_regressor_model.pkl'
params_path = 'rf_hyperparameters.json'

print("Cargando modelos e hiperparámetros...")
clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None

if os.path.exists(params_path):
    with open(params_path, 'r') as f:
        hyperparams = json.load(f)
        print("Hiperparámetros cargados exitosamente.")

# Crear directorio de salidas para las gráficas
output_dir = "shap_results_rf"
os.makedirs(output_dir, exist_ok=True)
print(f"Las gráficas se guardarán en la carpeta: {output_dir}/")

# ==============================================================================
# 3. SHAP PARA EL CLASIFICADOR (¿Es superconductor?)
# ==============================================================================
if clf_model is not None and y_cls.nunique() > 1:
    print("\n[1/2] Calculando valores SHAP para el CLASIFICADOR (Random Forest)...")
    
    # TreeExplainer optimizado para árboles de Scikit-Learn
    explainer_clf = shap.TreeExplainer(clf_model)
    
    # OPTIMIZACIÓN CPU: Tomamos una muestra de 5,000 filas para no saturar la memoria RAM
    X_sample_clf = X if len(X) <= 5000 else X.sample(n=5000, random_state=42)
    
    # Calculando valores
    shap_values_clf = explainer_clf(X_sample_clf)

    # Scikit-Learn Random Forest Classifier devuelve valores SHAP para ambas clases (0 y 1).
    # Filtramos para quedarnos solo con la explicación de la clase 1 (Es Superconductor).
    # En la API de SHAP actual, esto se hace seleccionando el índice [:, :, 1]
    if len(shap_values_clf.shape) == 3:
        shap_values_clf_pos = shap_values_clf[:, :, 1]
    else:
        shap_values_clf_pos = shap_values_clf

    # 3.1 Gráfico de Resumen (Beeswarm Plot) - Clasificador
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_clf_pos, max_display=15, show=False)
    plt.title("SHAP Beeswarm - Impacto en la Clasificación (Superconductor)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_rf_clf_beeswarm.png"), dpi=300)
    plt.close()

    # 3.2 Gráfico de Importancia Global (Bar Plot) - Clasificador
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_clf_pos, max_display=15, show=False)
    plt.title("Importancia Global de Características - Clasificador RF", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_rf_clf_bar.png"), dpi=300)
    plt.close()

    print(" -> Gráficos del clasificador generados y guardados con éxito.")
else:
    print("\n[1/2] Omitiendo clasificador (modelo 'rf_classifier_model.pkl' no encontrado).")

# ==============================================================================
# 4. SHAP PARA EL REGRESOR (Predicción de Temperatura Crítica)
# ==============================================================================
if reg_model is not None:
    print("\n[2/2] Calculando valores SHAP para el REGRESOR (Random Forest)...")
    
    # Filtramos los datos para usar SOLO los superconductores reales (como en el entrenamiento)
    X_supercon = X[y_cls == 1].reset_index(drop=True)
    
    explainer_reg = shap.TreeExplainer(reg_model)
    
    # OPTIMIZACIÓN CPU: Muestreo a 5000 para evitar overhead de memoria
    X_sample_reg = X_supercon if len(X_supercon) <= 5000 else X_supercon.sample(n=5000, random_state=42)
    
    # Calculando valores
    shap_values_reg = explainer_reg(X_sample_reg)

    # 4.1 Gráfico de Resumen (Beeswarm Plot) - Regresor
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_reg, max_display=15, show=False)
    plt.title("SHAP Beeswarm - Impacto en la Temp. Crítica (Regresión RF)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_rf_reg_beeswarm.png"), dpi=300)
    plt.close()

    # 4.2 Gráfico de Importancia Global (Bar Plot) - Regresor
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_reg, max_display=15, show=False)
    plt.title("Importancia Global de Características - Regresor RF", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_rf_reg_bar.png"), dpi=300)
    plt.close()
    
    # 4.3 Gráfico de Dependencia para la variable más importante
    # Busca la feature con mayor impacto medio absoluto
    if len(shap_values_reg.values) > 0:
        top_feature_idx = np.argsort(-np.abs(shap_values_reg.values).mean(0))[0]
        top_feature_name = X_sample_reg.columns[top_feature_idx]
        
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(shap_values_reg[:, top_feature_name], color=shap_values_reg, show=False)
        plt.title(f"Dependencia SHAP: {top_feature_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_rf_reg_dependence.png"), dpi=300)
        plt.close()

    print(" -> Gráficos del regresor generados y guardados con éxito.")
else:
    print("\n[2/2] Omitiendo regresor (modelo 'rf_regressor_model.pkl' no encontrado).")

print("\n" + "="*60)
print("¡ANÁLISIS SHAP FINALIZADO!")
print(f"Revisa la carpeta '{output_dir}' en tu directorio actual para ver las imágenes.")
print("="*60)