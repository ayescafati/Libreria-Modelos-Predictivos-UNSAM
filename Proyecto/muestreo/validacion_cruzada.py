import pandas as pd
from numpy.random import randint


def subconjuntos_aleatorios(df, k, agregar_restantes, semilla): 
    dimension_subconjunto = len(df) // k # k es el numero de subconjuntos o particiones en los que se divide el conjunto de datos original para el proceso de validacion cruzada

    subconjuntos = []
    for i in range(k):
        muestra = df.sample(n=dimension_subconjunto, random_state=semilla)
        df = df.drop(muestra.index, errors = 'ignorar')
        subconjuntos.append(muestra)

    if agregar_restantes: # agregamos los elementos restantes a los subconjuntos
        for i in range(len(df)):
            subconjuntos[i] = pd.concat([subconjuntos[i], df.iloc[i:i+1]])

    return subconjuntos


def generar_k_subconjuntos(df, k, tipo_muestreo = 'estratificado', agregar_restantes = True, semilla = randint(10000)):
    if tipo_muestreo == 'aleatorio':
        return subconjuntos_aleatorios(df, k, agregar_restantes, semilla)
    elif tipo_muestreo == 'estratificado':
        return estratificacion_de_los_subconjuntos(df, k, agregar_restantes, semilla)
    else:
        raise Exception("El parametro de muestreo debe ser estratificado o aleatorio.)
