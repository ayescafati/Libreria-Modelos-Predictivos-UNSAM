import pandas as pd
from numpy.random import randint
from typing import List, Tuple

def subconjuntos_aleatorios(df: pd.DataFrame, k: int, agregar_restantes: bool, semilla: int) -> List[pd.DataFrame]:
    dimension_subconjunto = len(df) // k # k es el numero de subconjuntos o particiones en los que se divide el conjunto de datos original para el proceso de validacion cruzada.

    subconjuntos = []
    for i in range(k):
        muestra = df.sample(n = dimension_subconjunto, random_state=semilla)
        df = df.drop(muestra.index, errors = 'ignorar')
        subconjuntos.append(muestra)

    if agregar_restantes: # Agregamos los elementos restantes a los subconjuntos.
        for i in range(len(df)):
            subconjuntos[i] = pd.concat([subconjuntos[i], df.iloc[i:i+1]])

    return subconjuntos


def estratificacion_de_los_subconjuntos(df: pd.DataFrame, k: int, agregar_restantes: bool, semilla: int) -> List[pd.DataFrame]: # La estratificación es aquella técnica de muestreo que se utiliza para garantizar que las proporciones de las clases en los datos de entrenamiento y prueba sean lo más similares posibles.
    grupos = df.groupby('clase')
    subconjutons_por_grupo = [subconjuntos_aleatorios(grupo, k, agregar_restantes, semilla) for valor_clase, grupo in grupos]
    subconjuntos = [pd.concat(subconjutons_por_grupo[x][y] for x in range(len(subconjutons_por_grupo))) for y in range(k)]
    return subconjuntos


def generar_k_subconjuntos(df: pd.DataFrame, k: int, tipo_muestreo: str = 'estratificado', agregar_restantes: bool = True, semilla: int = randint(0, 10000)) -> List[pd.DataFrame]:
    if tipo_muestreo == 'aleatorio':
        return subconjuntos_aleatorios(df, k, agregar_restantes, semilla)
    elif tipo_muestreo == 'estratificado':
        return estratificacion_de_los_subconjuntos(df, k, agregar_restantes, semilla)
    else:
        raise Exception("El parametro de muestreo debe ser uno de los siguientes: [estratificado, aleatorio]")
