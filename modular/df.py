import pandas as pd
import numpy as np

df1 = pd.read_csv("SuperCon_MTG.csv")
a_1 = df1.head()

df2 = pd.read_csv("optimized_supercon.csv")
a_2 = df2.head()

print(a_1)
print("\n" * 3)
print(a_2)


b_1 = df1.shape
b_2 = df2.shape
print(f"Las dimensiones pasaron de {b_1} a {b_2}")