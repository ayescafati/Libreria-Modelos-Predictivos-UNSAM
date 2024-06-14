

import statistics
import numpy as np


def registrar(texto, mostrar_detalle_adicional=False):
    if mostrar_detalle_adicional:
        print(texto)


def obtener_valores_posibles(dataset, atributo):
    assert not isinstance(dataset[0][atributo], float)
    divisiones = set()
    for instance in dataset:
        valor_atributo = instance[atributo]
        divisiones.add(valor_atributo)

    return divisiones


def dividir_en_partes(secuencia, num_partes):
    tamano_promedio = len(secuencia) / float(num_partes)
    partes = []
    ultimo = 0.0

    while ultimo < len(secuencia):
        partes.append(secuencia[int(ultimo):int(ultimo + tamano_promedio)])
        ultimo += tamano_promedio

    return partes


def calcular_exactitud(etiquetas_predichas, etiquetas_reales):
    etiquetas_predichas = np.asarray(etiquetas_predichas)
    etiquetas_reales = np.asarray(etiquetas_reales)
    correctas = np.sum(etiquetas_predichas == etiquetas_reales)
    exactitud = correctas / len(etiquetas_reales)
    assert 0 <= exactitud <= 1
    return exactitud


def calcular_precision_clase(etiquetas_predichas, etiquetas_reales):
    assert len(etiquetas_predichas) == len(etiquetas_reales)
    clases = set(etiquetas_reales)
    precisions = []
    etiquetas_predichas = np.asarray(etiquetas_predichas)
    etiquetas_reales = np.asarray(etiquetas_reales)
    for clase in clases:  # Calcula la precisión para cada clase
        precision_clase_iterada = etiquetas_predichas == clase
        if sum(precision_clase_iterada) == 0:
            continue
        cant_instancias_bien_predichas = 0 # cant_instancias_bien_predichas es la cantidad de instancias correctamente predichas para la clase sobre la cual estamos iterando.
        for etiqueta_predicha, etiqueta_real in zip(etiquetas_predichas[precision_clase_iterada], etiquetas_reales[precision_clase_iterada]):
            if etiqueta_predicha == etiqueta_real:
                cant_instancias_bien_predichas += 1
        precisiones_clase = cant_instancias_bien_predichas / np.sum(precision_clase_iterada)
        assert 0 <= precisiones_clase <= 1
        precisions.append(precisiones_clase)

    return statistics.mean(precisions)


def recall(etiquetas_predichas, etiquetas_reales) -> float: # devuelve el valor del recall para el modelo
    '''
    Calcula el recall para un problema de clasificación.

    El recall (o tasa de verdaderos positivos) es una métrica de evaluación que mide la capacidad de un modelo 
    para identificar todas las muestras positivas en un conjunto de datos.
    '''
    clases_unicas = set(etiquetas_reales)
    recalls_por_clase = [ ]
    etiquetas_predichas = np.asarray(etiquetas_predichas)
    etiquetas_reales = np.asarray(etiquetas_reales)
    for clase in clases_unicas:  

        relevant = etiquetas_reales == clase
        cant_instancias_bien_predichas = 0
        for etiqueta_predicha, etiqueta_real in zip(etiquetas_predichas[relevant], etiquetas_reales[relevant]):
            if etiqueta_predicha == etiqueta_real:
                cant_instancias_bien_predichas += 1

        recall_para_esta_clase = cant_instancias_bien_predichas / np.sum(relevant) 
        assert 0 <= recall_para_esta_clase <= 1
        recalls_por_clase.append(recall_para_esta_clase)

    return statistics.mean(recalls_por_clase)  # Devuelve el promedio de los recalls de todas las clases. Esto es, el valor del recall para el modelo


def f1_score(etiquetas_predichas, etiquetas_reales):
    precision = calcular_precision_clase(etiquetas_predichas, etiquetas_reales)
    recll = recall(etiquetas_predichas, etiquetas_reales)

    score = statistics.median([precision, recll])
    assert 0 <= score <= 1
    return score
