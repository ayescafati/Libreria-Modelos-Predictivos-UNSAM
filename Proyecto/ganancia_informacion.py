
from scipy import stats
import numpy as np
import math


def info(clases) -> float:
    '''
    Recibe una lista con las clases y calcula la entropía.
    La entropía es una medida de la incertidumbre en un conjunto de datos.
    '''
    # 'clases' es una lista con las etiquetas de clase
    # 'clases_vals' es un set con los valores únicos de 'clases'
    clases_vals = set(clases)
    sum = 0

    # Para cada valor único en 'clases_vals', calculamos la probabilidad (p_i) de que
    # aparezca ese valor en 'clases', y después sumamos la contribución de ese valor
    # a la entropía total usando la fórmula de la entropía:
    # Entropía = -sum(p_i * log2(p_i)) para cada clase en 'clases_vals'
    for n in clases_vals:
        pn = clases.count(n) / len(clases)
        sum += -pn * math.log(pn, 2)

    return sum   # Devolvemos la entropía calculada


def info_numerica(dataset, atributo, division) -> float:
    clase_menor = [x['class'] for x in dataset if x[atributo] < division]
    clase_mayor = [x['class'] for x in dataset if x[atributo] > division]

    return (len(clase_menor) / len(dataset)) * info(clase_menor) + (
                len(clase_mayor) / len(dataset)) * info(clase_mayor)


def calcular_ganancia_info_num(dataset, atributo) -> tuple: # devuelve una una tupla que contiene la ganancia de información y el mejor valor de division
    '''
    Calcula la ganancia de información para un atributo numérico en un conjunto de datos.
    La ganancia de información mide cuánto se reduce la incertidumbre sobre la clase
    al conocer el valor de un atributo específico.
    '''

    # Sacamos los valores únicos del atributo en cuestión
    valores = {x[atributo] for x in dataset}  # Crea un conjunto para eliminar duplicados valores = lista(valores). Esto es, para quedarnos con los valores unicos.
    valores = list(valores)
    if len(valores) == 1:  # Si el atributo es puro (o sea, si tiene un solo valor). No tiene sentido dividirlo
        return 0, None

    # Calculamos los puntos medios entre cada par de valores únicos consecutivos
    divisiones_medias =  [(valor1 + valor2) / 2 for valor1, valor2 in zip(valores[:-1], valores[1:])]

    # Calculamos la ganancia de información para cada punto de división
    ganancias_de_info = [info_numerica(dataset, atributo, punto) for punto in divisiones_medias]

    # Encontramos el índice de la mejor división (la que tiene la mayor ganancia de información)
    indice_mejor_division = np.argmax(np.asarray(ganancias_de_info))
    mejor_valor_division = divisiones_medias[indice_mejor_division]

    # Calculamos la entropía inicial del dataset
    clases = [x['class'] for x in dataset]
    information = info(clases)

    # Dividimos el conjunto de datos en dos subconjuntos basados en la mejor división encontrada
    clase_menor = [x['class'] for x in dataset if x[atributo] < mejor_valor_division]
    clase_mayor = [x['class'] for x in dataset if x[atributo] > mejor_valor_division]

    # Calculamos la entropía ponderada de los subconjuntos
    informacion_despues_de_division = (len(clase_menor) / len(dataset)) * info(clase_menor) + (
                len(clase_mayor) / len(dataset)) * info(clase_mayor)

    # Luego, la ganancia de información es la diferencia entre la entropía original y la entropía de los subconjuntos
    return information - informacion_despues_de_division, mejor_valor_division


def calcular_ganancia_info_cat(dataset, atributo) -> float: # devuelve la ganancia de información
    '''
    Este método nos dice cuánto beneficio sacás de un atributo que divide tus datos.
    '''

    # Armamos un diccionario para ver cómo quedan las divisiones (conjuntos) según el valor del atributo
    divisiones_por_valor = {}
    for instancia in dataset:
        valor_atributo = instancia[atributo]
        if valor_atributo not in divisiones_por_valor.keys():
            divisiones_por_valor[valor_atributo] = [instancia]
        else:
            divisiones_por_valor[valor_atributo].append(instancia)

    # Sacamos el total de instancias
    cantidad_total_instancias = len(dataset)  # Numero total de instancias en el dataset

    # Calculamos la incertidumbre inicial del conjunto de datos
    clases = [ dato ['class'] for dato in dataset ]
    informacion_inicial = info(clases)

    # Ahora calculamos cuánta incertidumbre queda después de la división
    informacion_despues_de_division = 0
    for valor_atributo in divisiones_por_valor.keys():
        # Sacamos cuántas instancias hay en este conjunto
        cantidad_instancias_grupo = len(divisiones_por_valor[valor_atributo])   # Numero de instancias en esta clase
        clases = [x['class'] for x in divisiones_por_valor[valor_atributo]]

        informacion_despues_de_division += (cantidad_instancias_grupo / cantidad_total_instancias) * info(clases)

    return informacion_inicial - informacion_despues_de_division
