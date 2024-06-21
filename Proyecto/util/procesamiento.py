################ BLOQUE DE IMPORTACIONES ################
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
import random
import time

from metricas.matriz_confusion import MatrizConfusion
from util.funcionalidades import cargar_datos, cargar_atributos, guardar_resultados
from muestreo import generar_k_subconjuntos, dividir_entrenamiento_y_pruebas
from modelos.random_forest import RandomForest
########################################################

def procesar_dataset(nombre_dataset, ruta_dataset, ruta_atributos, semilla, cardinal_k_subconjuntos, numero_arboles, nivel_verbosidad, paralelizar):
    datos = cargar_datos(ruta_dataset)
    atributos = cargar_atributos(ruta_atributos)
    atributos_por_division = int(round(np.sqrt(len(atributos))))


    print("=" * 90)
    print(f"Resultados para {nombre_dataset}:")
    print(
        f"Parámetros usados: \n Cantidad de Subconjuntos: {cardinal_k_subconjuntos} \n Número de árboles: {numero_arboles} \n Número de atributos: {atributos_por_division} \n Semilla: {semilla}")

    if semilla and nivel_verbosidad > 0:
        random.seed(semilla)

    k_subconjuntos = generar_k_subconjuntos(datos, cardinal_k_subconjuntos, semilla=semilla)
    divisiones = dividir_entrenamiento_y_pruebas(k_subconjuntos)

    if paralelizar:
        pool = Pool(os.cpu_count() * 2 - 1)
    else:
        pool = None

    resultados_totales = []
    tiempo_inicio_total = time.time()
    for i, division in enumerate(divisiones):
        entrenado, test = division

        arrancar = time.time()
        forest = RandomForest.entrenar(entrenado, atributos, numero_arboles, numero_atributos=atributos_por_division, pool=pool, semilla=semilla)
        terminar = time.time()

        if nivel_verbosidad > 1:
            print("=" * 90)
            print(f"Random Forest {i + 1} tiempo de creación: {terminar - arrancar:.3f} segundos")

        resultados = forest.salida_predicciones(test)
        resultados_totales.append(resultados)

        if nivel_verbosidad > 1:
            matriz_confusion = MatrizConfusion(resultados)
            matriz_confusion.mostrar(nivel_verbosidad=(nivel_verbosidad > 2))

    final_matriz_confusion = MatrizConfusion(pd.concat(resultados_totales))
    tiempo_final_total = time.time()

    final_matriz_confusion.mostrar(nivel_verbosidad=(nivel_verbosidad > 0))
    tiempo_de_ejecucion = tiempo_final_total - tiempo_inicio_total
    print(f"Tiempo total de procesamiento: {tiempo_de_ejecucion:.3f} segundos")

    carpeta_predicciones = "predicciones"
    if not os.path.exists(carpeta_predicciones):
        os.makedirs(carpeta_predicciones)

    nuevo_nombre_archivo = f"{nombre_dataset}_predicciones.csv"
    ruta_predicciones = os.path.join(carpeta_predicciones, nuevo_nombre_archivo)

    resultados_totales_df = pd.concat(resultados_totales)
    resultados_totales_df.to_csv(ruta_predicciones, index=False)

    print("=" * 90)
    print(f"Predicciones guardadas en '{ruta_predicciones}'")

    guardar_resultados(
        final_matriz_confusion,
        ruta_dataset,
        cardinal_k_subconjuntos,
        numero_arboles,
        atributos_por_division,
        tiempo_de_ejecucion,
        semilla,
        paralelizar)

    if paralelizar:
        pool.close()
