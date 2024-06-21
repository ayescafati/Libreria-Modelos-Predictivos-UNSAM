from typing import List, Tuple, Union
from numba import jit
import numpy as np


@jit(nopython = True)
def entropia(ocurrencias_clase: np.ndarray, total_ocurrencias: int) -> float:
    total_info: float = 0.0
    for ocurrencia in ocurrencias_clase:
        probabilidad: float = ocurrencia / total_ocurrencias
        total_info -= probabilidad * np.log2(probabilidad)
    return total_info

def agrupar_por_atributo(data: np.ndarray, indice_atributos: List[int], atributo: int, tipo_cada_atributo: str) -> Tuple[List[np.ndarray], Union[List[int], List[bool]]]:
    if tipo_cada_atributo == "nominal":
        indice_de_la_particion: np.ndarray = np.unique(data[:, indice_atributos[atributo]])
        grupos: List[np.ndarray] = [data[data[:, indice_atributos[atributo]] == i] for i in indice_de_la_particion]
        return grupos, indice_de_la_particion
    else:
        columna: np.ndarray = data[:, indice_atributos[atributo]]
        mean: float = columna.mean()
        mayor_que: np.ndarray = data[data[:, indice_atributos[atributo]] <= mean]
        menor_que: np.ndarray = data[data[:, indice_atributos[atributo]] > mean]

        if not menor_que.any():
            return [mayor_que], [False]
        elif not mayor_que.any():
            return [menor_que], [True]
        else:
            return [mayor_que, menor_que], [False, True]
        

def info_atributos(data: np.ndarray, indice_atributos: List[int], tipo_atributos: List[str], lista_atributos: List[Tuple[int, str]], indice_clases: int) -> Tuple[List[float], List[List[np.ndarray]], List[Union[List[int], List[bool]]]]:
    dimension_total: int = data.shape[0]
    infos: List[float] = []
    todos_los_grupos: List[List[np.ndarray]] = []
    indice_todos_los_grupos: List[Union[List[int], List[bool]]] = []

    for atributo, tipo_cada_atributo in lista_atributos:
        grupos, indice_grupos = agrupar_por_atributo(data, indice_atributos, atributo, tipo_cada_atributo)
        atributo_info: float = 0.0
        for grupo in grupos:
            dimension_grupo: int = len(grupo)
            clase_columna: np.ndarray = grupo[:, indice_clases]
            clases, counts = np.unique(clase_columna, return_counts=True)
            atributo_info += dimension_grupo / dimension_total * entropia(counts, dimension_grupo)
        
        infos.append(atributo_info)
        todos_los_grupos.append(grupos)
        indice_todos_los_grupos.append(indice_grupos)

    return infos, todos_los_grupos, indice_todos_los_grupos

def atributos_ganancia(data: np.ndarray, indice_atributos: List[int], tipo_atributos: List[str], lista_atributos: List[Tuple[int, str]], indice_clases: int) -> Tuple[List[float], List[List[np.ndarray]], List[Union[List[int], List[bool]]]]:
    dimension_total: int = data.shape[0]
    clases, counts = np.unique(data[:, indice_clases], return_counts=True)
    df_info: float = entropia(counts, dimension_total)
    atributo_infos, todos_los_grupos, indice_todos_los_grupos = info_atributos(data, indice_atributos, tipo_atributos, lista_atributos, indice_clases)
    ganancia: List[float] = [df_info - info for info in atributo_infos]
    return ganancia, todos_los_grupos, indice_todos_los_grupos
