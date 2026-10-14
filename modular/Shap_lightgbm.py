import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings

# Suprimir advertencias menores para consola limpia
warnings.filterwarnings("ignore")

# Configurar el número de hilos (threads) para maximizar el uso del Intel i5-1135G7 (8 hilos)
# Esto acelera operaciones subyacentes de C++ utilizadas por SHAP y LightGBM
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

# Desactivar interactividad de pyplot para guardar gráficos rápidamente sin bloquear la RAM/CPU
plt.ioff()
# Usar un estilo limpio para los gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("=" * 60)
print("INICIANDO ANÁLISIS DE EXPLICABILIDAD CON SHAP")
print("Procesador: Intel Core i5-1135G7 (Optimizaciones multi-hilo activas)")
print("=" * 60)

# 1. Cargar el Dataset
dataset_path = "optimized_supercon.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"No se encontró el archivo '{dataset_path}'.")

print("Cargando dataset...")
df = pd.read_csv(dataset_path)
X = df.drop(columns=['Temp_critical'])
y_reg = df['Temp_critical']
y_cls = (y_reg > 0).astype(int)

# 2. Cargar Modelos
clf_path = 'lgbm_classifier.pkl'
reg_path = 'lgbm_regressor.pkl'

clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None

# Crear directorio de salidas para las gráficas
output_dir = "shap_results"
os.makedirs(output_dir, exist_ok=True)
print(f"Las gráficas se guardarán en la carpeta: {output_dir}/")

# ==============================================================================
# 3. SHAP PARA EL CLASIFICADOR (¿Es superconductor?)
# ==============================================================================
if clf_model is not None and y_cls.nunique() > 1:
    print("\n[1/2] Calculando valores SHAP para el CLASIFICADOR...")
    
    # TreeExplainer de SHAP está altamente optimizado en C++ para árboles (LightGBM)
    explainer_clf = shap.TreeExplainer(clf_model)
    
    # OPTIMIZACIÓN CPU: Si el dataset es masivo, calculamos SHAP en una muestra aleatoria
    # de 5,000 filas. Esto mantiene la precisión analítica pero evita saturar la RAM del i5.
    X_sample_clf = X if len(X) <= 5000 else X.sample(n=5000, random_state=42)
    
    # Calculando valores
    shap_values_clf = explainer_clf(X_sample_clf)

    # 3.1 Gráfico de Resumen (Beeswarm Plot) - Clasificador
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_clf, max_display=15, show=False)
    plt.title("SHAP Beeswarm - Impacto en la Clasificación (Superconductor vs No)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_clf_beeswarm.png"), dpi=300)
    plt.close()

    # 3.2 Gráfico de Importancia Global (Bar Plot) - Clasificador
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_clf, max_display=15, show=False)
    plt.title("Importancia Global de Características - Clasificador", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_clf_bar.png"), dpi=300)
    plt.close()

    print(" -> Gráficos del clasificador generados y guardados con éxito.")
else:
    print("\n[1/2] Omitiendo clasificador (modelo no encontrado o dataset de una sola clase).")

# ==============================================================================
# 4. SHAP PARA EL REGRESOR (Predicción de Temperatura Crítica)
# ==============================================================================
if reg_model is not None:
    print("\n[2/2] Calculando valores SHAP para el REGRESOR...")
    
    # Filtramos los datos que usamos para la regresión (solo los superconductores reales)
    X_supercon = X[y_cls == 1].reset_index(drop=True)
    
    explainer_reg = shap.TreeExplainer(reg_model)
    
    # OPTIMIZACIÓN CPU: Muestreo a 5000 para el regresor si es muy grande
    X_sample_reg = X_supercon if len(X_supercon) <= 5000 else X_supercon.sample(n=5000, random_state=42)
    
    # Calculando valores
    shap_values_reg = explainer_reg(X_sample_reg)

    # 4.1 Gráfico de Resumen (Beeswarm Plot) - Regresor
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_reg, max_display=15, show=False)
    plt.title("SHAP Beeswarm - Impacto en la Temp. Crítica (Regresión)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_reg_beeswarm.png"), dpi=300)
    plt.close()

    # 4.2 Gráfico de Importancia Global (Bar Plot) - Regresor
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_reg, max_display=15, show=False)
    plt.title("Importancia Global de Características - Regresor", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_reg_bar.png"), dpi=300)
    plt.close()
    
    # 4.3 Gráfico de Dependencia para la variable más importante (opcional pero muy útil)
    # Encuentra la feature más importante según SHAP para graficarla
    top_feature_idx = np.argsort(-np.abs(shap_values_reg.values).mean(0))[0]
    top_feature_name = X_sample_reg.columns[top_feature_idx]
    
    plt.figure(figsize=(8, 6))
    shap.plots.scatter(shap_values_reg[:, top_feature_name], color=shap_values_reg, show=False)
    plt.title(f"Dependencia SHAP: {top_feature_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_reg_dependence.png"), dpi=300)
    plt.close()

    print(" -> Gráficos del regresor generados y guardados con éxito.")
else:
    print("\n[2/2] Omitiendo regresor (modelo 'lgbm_regressor.pkl' no encontrado).")

print("\n" + "="*60)
print("¡ANÁLISIS SHAP FINALIZADO!")
print(f"Revisa la carpeta '{output_dir}' para ver tus gráficas de explicabilidad.")
print("="*60)