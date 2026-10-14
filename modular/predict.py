
import joblib
import pandas as pd

from pymatgen.core import Composition
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    BandCenter,
    ValenceOrbital,
)

# --------------------------------------------------
# 1. Compuesto de prueba
# --------------------------------------------------
formula = "H3S"
composition = Composition(formula)

# --------------------------------------------------
# 2. Generar las features con los mismos featurizers
# --------------------------------------------------
data = pd.DataFrame({"composition": [composition]})

featurizers = [
    ElementProperty.from_preset("magpie"),
    Stoichiometry(),
    BandCenter(),
    ValenceOrbital(props=["frac"]),
]

for featurizer in featurizers:
    data = featurizer.featurize_dataframe(
        data,
        col_id="composition",
        ignore_errors=False,
    )

X_new = data.drop(columns=["composition"])

# --------------------------------------------------
# 3. Alinear las features con las del entrenamiento
# --------------------------------------------------
# Se usa el CSV como plantilla de columnas.
# No se utiliza Temp_critical para generar la predicción.
template = pd.read_csv("optimized_supercon.csv", nrows=0)

feature_cols = [
    col for col in template.columns
    if col != "Temp_critical"
]

missing = sorted(set(feature_cols) - set(X_new.columns))
extra = sorted(set(X_new.columns) - set(feature_cols))

if missing or extra:
    raise ValueError(
        f"Las features no coinciden.\n"
        f"Faltantes: {missing}\n"
        f"Adicionales: {extra}\n"
        "Revisa los nombres y versiones de los featurizers."
    )

X_new = X_new.reindex(columns=feature_cols)

if X_new.isna().any().any():
    raise ValueError("Hay valores NaN en las features del compuesto.")

# --------------------------------------------------
# 4. Cargar los modelos YA entrenados
# --------------------------------------------------
models = {
    "LightGBM": joblib.load("lightgbm_model.pkl"),
    "CatBoost": joblib.load("catboost_model.pkl"),

}

# --------------------------------------------------
# 5. Predecir Tc
# --------------------------------------------------
print(f"Compuesto: {formula}")
print("Predicciones de temperatura crítica:\n")

for name, model in models.items():
    prediction = float(model.predict(X_new)[0])
    print(f"{name:15s}: {prediction:.2f} K")
