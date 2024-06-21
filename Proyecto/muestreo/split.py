################ BLOQUE DE IMPORTACIONES ################
import pandas as pd
from typing import List, Tuple
#######################################################

def dividir_entrenamiento_y_pruebas(subconjuntos: List[pd.DataFrame]) -> List[Tuple[pd.DataFrame, pd.DataFrame]]: #divide el dataset en dos conjuntos, uno de entrenamiento y otro de pruebas
    conjuntos = []
    for i, subconjunto in enumerate(subconjuntos):
        conjuntos.append((pd.concat(subconjuntos[:i] + subconjuntos[i + 1:]), subconjuntos[i]))

    return conjuntos
