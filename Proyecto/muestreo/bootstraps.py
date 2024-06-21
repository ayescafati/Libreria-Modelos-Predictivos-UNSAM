################ BLOQUE DE IMPORTACIONES ################
from numpy.random import randint
from typing import List
import pandas as pd 
#########################################################

def generar_bootstraps(df: pd.DataFrame, numero_muestras: int, semilla: int = randint(0, 10000)) -> List[pd.DataFrame]:
    bootstraps = []
    for i in range(numero_muestras):
        muestra = df.sample(frac=1, replace = True, random_state=semilla+i) # en Matemática y/o Estadística computacional entendemos por "semilla" a aquel nuúero (o vector) utilizado para inicializar un generador de números pseudoaleatorios
        bootstraps.append(muestra)

    return bootstraps
