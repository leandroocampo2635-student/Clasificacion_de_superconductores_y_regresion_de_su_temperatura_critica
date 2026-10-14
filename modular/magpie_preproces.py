
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
# 1. Load dataset
df = pd.read_csv('SuperCon_MTG.csv')

# 2. Define robust parser
def safe_composition(formula):
    try:
        return Composition(formula)
    except Exception:
        return None

# 3. Apply parsing and filter out invalid rows
df['composition'] = df['Chemical_Formula'].apply(safe_composition)

# Count and report removed formulas
invalid_count = df['composition'].isna().sum()
print(f"Removed {invalid_count} invalid chemical formula(s).")

# Keep only valid rows
df = df.dropna(subset=['composition']).reset_index(drop=True)

# 4. Proceed with feature extraction
magpie = ElementProperty.from_preset("magpie")
df = magpie.featurize_dataframe(df, col_id='composition')

stoich = Stoichiometry()
valence = ValenceOrbital()
band = BandCenter()

print("Extracting stoichiometry & symmetry features...")
df = stoich.featurize_dataframe(df, col_id='composition')

print("Extracting valence orbital proportions...")
df = valence.featurize_dataframe(df, col_id='composition')

print("Extracting energy band centers...")
df = band.featurize_dataframe(df, col_id='composition')

# 5. Clean up temporary composition column
df = df.drop(columns=['composition'])

# Save the featurized dataset

print(f"Dataset successfully featurized! Output shape: {df.shape}")

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