from collections import Counter
from numpy.random import randint as entero_aleatorio

from muestreo import generar_bootstraps
from modelos import ArbolDecision, predecir as arbol_predecir

import pandas as pd
from typing import List, Optional, Tuple, Any
from random import randint
from collections import Counter
from multiprocessing.pool import Pool


class RandomForest:
    def __init__(self, arboles):
        self.arboles = arboles

    @classmethod
    def entrenar(cls, conjunto_entrenamiento: pd.DataFrame, atributos: List[str], numero_arboles: int, numero_atributos: Optional[int] = None, pool: Optional[Pool] = None, semilla: Optional[int] = None) -> 'RandomForest':
        if not semilla:
            semilla = entero_aleatorio(10000)
        bootstraps = generar_bootstraps(conjunto_entrenamiento, numero_arboles, semilla= semilla)
        combinaciones_bootstraps_atributos = [(b, atributos, numero_atributos) for b in bootstraps]
        if pool:
            arboles = pool.starmap(ArbolDecision.entrenar, combinaciones_bootstraps_atributos)
        else:
            arboles = [ArbolDecision.entrenar(b, atributos, numero_atributos) for b in bootstraps]

        return RandomForest(arboles)
    

    def predecir(self, observaciones: pd.Series, pool = None) -> Any:
        instancias_arboles = [(arbol, observaciones) for arbol in self.arboles]

        if pool:
            resultados = pool.starmap(arbol_predecir, instancias_arboles)

        else:
            resultados = [arbol_predecir(arbol, observaciones) for arbol in self.arboles]

        datos = Counter(resultados)
        resultado = max(resultados, key = datos.get)

        return resultado

    def salida_predicciones(self, observaciones: pd.DataFrame) -> 'pd.DataFrame':
        observaciones['prediccion'] = observaciones.apply(lambda x: self.predecir(x), axis = 1)
        return observaciones[['clase', 'prediccion']]
