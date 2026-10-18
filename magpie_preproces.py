'''Ejecutado en google colab con la version de python 3.13 debido a problemas con python 3.14 de mi computadora local

'''

import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import (
  ElementProperty,
  Stoichiometry,
  ValenceOrbital,
  BandCenter
)
import json
import numpy as np
# 1. Carga del dataset SuperCon_MTG.csv
df = pd.read_csv('SuperCon_MTG.csv')

''' 2. Define una funcion que asegura que todas las composiciones son validas
para el proceso magpie
'''
def safe_composition(formula):
    try:
        return Composition(formula)
    except Exception:
        return None

# 3. Se aplica la funcion a todas las composicones de la base de datos
df['composition'] = df['Chemical_Formula'].apply(safe_composition)

# Cuenta y remueve las composiciones que no cumplan los requerimientos del proceso magpie como indices negativos
invalid_count = df['composition'].isna().sum()
print(f"Removed {invalid_count} invalid chemical formula(s).")

# Mantiene solo las filas que si cumplen los requerimientos
df = df.dropna(subset=['composition']).reset_index(drop=True)

# 4. Proceso de feautrizacion
magpie = ElementProperty.from_preset("magpie")
df = magpie.featurize_dataframe(df, col_id='composition')

stoich = Stoichiometry()
valence = ValenceOrbital()
band = BandCenter()

print("Extrayendo estequimetria y caracteristicas de simetría...")
df = stoich.featurize_dataframe(df, col_id='composition')

print("Extrayendo proporciones de orbitales de valencia...")
df = valence.featurize_dataframe(df, col_id='composition')

print("Extrayendo centros de bandas...")
df = band.featurize_dataframe(df, col_id='composition')

# 5. Se remueve la columna composicion
df = df.drop(columns=['composition'])

# Se salva

print(f"Dataset exitosamente featurizado, tamaño del dataset: {df.shape}")
print("Salvando dataset como clean_supercon.csv")
df.to_csv('clean_supercon.csv', index=False)



# 1. Cargar datos
df = pd.read_csv("clean_supercon.csv")

# 2. Eliminar identificadores que causan Data Leakage
df_ml = df.drop(columns=['Material-ID', 'Chemical_Formula'])

# 3. Resolver colisiones (filas con mismas características pero distinta Temp_critical)
# Agrupamos por todas las características numéricas y promediamos la temperatura crítica
features = [c for c in df_ml.columns if c != 'Temp_critical']
df_ml_grouped = df_ml.groupby(features, as_index=False)['Temp_critical'].mean()

# 4. Eliminar Multicolinealidad (Variables redundantes con correlación > 0.95)
corr_matrix = df_ml_grouped.drop(columns=['Temp_critical']).corr().abs()
# Triángulo superior de la matriz de correlación
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Identificar columnas a descartar
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

df_final = df_ml_grouped.drop(columns=to_drop)

# 5. Guardar el dataset limpio y listo para Machine Learning
#df_final.to_csv("optimized_supercon.csv", index=False)

feature_cols = [
    c for c in df.columns
    if c != "Temp_critical"
]

with open("feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"Filas originales: {df.shape[0]}")
print(f"Filas tras resolver ruido: {df_ml_grouped.shape[0]}")
print(f"Columnas eliminadas (alta correlación): {len(to_drop)}")
print(f"Shape final del dataset: {df_final.shape}")