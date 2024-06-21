import os
import json
from time import time
import pandas as pd
import numpy as np
from typing import Any, Dict, Union

from metricas.matriz_confusion import MatrizConfusion

def cargar_datos(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df: pd.DataFrame = pd.read_csv(path)
        if 'Clase' in df.columns:
            df = df.rename(columns={'Clase': 'clase'})
        elif 'CLASE' in df.columns:
            df = df.rename(columns={'CLASE': 'clase'})

        if 'clase' not in df.columns:
            raise Exception("EL CONJUNTO DE DATOS NO CONTIENE LA COLUMNA \"clase\" ")

        return df
    else:
        raise Exception("EL ARCHIVO NO EXISTE")

def cargar_atributos(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, 'r') as archivo:
            return json.load(archivo)
    else:
        raise Exception("FALTA INFORMACION DE LA CATEGORIA DE LOS ATRIBUTOS")

def guardar_resultados(matriz_de_confusion: 'MatrizConfusion', ruta_de_los_datos: str, cantidad_k_subconjuntos: int, numero_arboles: int, numero_atributos: int, tiempo_ejecucion: float, semilla: Union[int, None], paralelizar: bool, path: str = 'resultados') -> None:
    nombre_dataset: str = os.path.basename(ruta_de_los_datos)
    file_path: str = os.path.join(path, nombre_dataset)

    if not os.path.exists(path):
        os.mkdir(path)

    existe_el_archivo: bool = os.path.exists(file_path)
    with open(file_path, 'a+') as f:
        if not existe_el_archivo:
            encabezado_del_archivo: str = '"total","correcto","exactitud","verdaderos_positivos","verdaderos_negativos","falsos_positivos","falsos_negativos","clases","recall_por_clase","precision_por_clase","especificidad_por_clase","macro_recall","macro_precision","macro_especificidad","macro_f_score_2","macro_f_score_1","macro_f_score_0.5","cantidad_k_subconjuntos","numero_arboles","numero_atributos","tiempo_de_ejecucion","timestamp","semilla"\r\n'
            f.write(encabezado_del_archivo)

        semilla = semilla if semilla and not paralelizar else 0
        clases: str = str(np.unique(matriz_de_confusion._resultados.values[:, 0]))

        total: int = matriz_de_confusion._total
        correcto: int = matriz_de_confusion._correcto
        exactitud: float = matriz_de_confusion.exactitud()

        verdaderos_positivos: str = f'"{str(matriz_de_confusion.verdaderos_positivos().to_list())}"'
        verdaderos_negativos: str = f'"{str(matriz_de_confusion.verdaderos_negativos().to_list())}"'
        falsos_positivos: str = f'"{str(matriz_de_confusion.falsos_positivos().to_list())}"'
        falsos_negativos: str = f'"{str(matriz_de_confusion.falsos_negativos().to_list())}"'
        recall_por_clase: str = f'"{str(matriz_de_confusion.recalls().to_list())}"'
        precision_por_clase: str = f'"{str(matriz_de_confusion.precision().to_list())}"'
        especificidad_por_clase: str = f'"{str(matriz_de_confusion.especificidad().to_list())}"'

        macro_recall: float = matriz_de_confusion.macro_recall()
        macro_precision: float = matriz_de_confusion.macro_precision()
        macro_especificidad: float = matriz_de_confusion.macro_especificidad()
        macro_f_score_2: float = matriz_de_confusion.macro_f_score(2)
        macro_f_score_1: float = matriz_de_confusion.macro_f_score(1)
        macro_f_score_05: float = matriz_de_confusion.macro_f_score(0.5)
        timestamp: int = int(time())

        resultados: str = f'{total},{correcto},{exactitud},{verdaderos_positivos},{verdaderos_negativos},{falsos_positivos},{falsos_negativos},{clases},{recall_por_clase},{precision_por_clase},{especificidad_por_clase},{macro_recall},{macro_precision},{macro_especificidad},{macro_f_score_2},{macro_f_score_1},{macro_f_score_05},{cantidad_k_subconjuntos},{numero_arboles},{numero_atributos},{tiempo_ejecucion},{timestamp},{semilla}\r\n'

        f.write(resultados)