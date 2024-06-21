################ BLOQUE DE IMPORTACIONES ################
from typing import Dict, Union, List, Tuple
import random
import numpy as np
from informacion import info_atributos
########################################################


class ArbolDecision:
    def __init__(self, opciones: Union[Dict, None] = None, etiqueta_clase: Union[int, None] = None,
                 atributo: Union[str, None] = None, tipo: Union[str, None] = None, valor_de_corte: Union[float, None] = None):
        self.etiqueta_clase = etiqueta_clase
        self.atributo = atributo
        self.valor_de_corte = valor_de_corte
        self.opciones = opciones
        self.tipo = tipo


    @classmethod
    def entrenar(cls, df: np.ndarray, tipo_atributos: Dict[str, str], numero_elementos: Union[int, None] = None) -> 'ArbolDecision':
        indice_atributos = {valor: i for i, valor in enumerate(df.columns.values)}
        indice_clases = indice_atributos.pop('clase')
        dato = df.values
        return _entrenar(dato, indice_atributos, tipo_atributos, indice_clases, numero_elementos)


def clase_mas_frecuente(clases: np.ndarray) -> int:
    valores, conteos = np.unique(clases, return_counts=True)
    return valores[conteos.argmax()]


def verificar_clase_unica(datos: np.ndarray, indice_clases: int) -> bool:
    return np.unique(datos[:, indice_clases]).size == 1


def seleccionar_numero_atributos(atributos: Dict[str, str], numero_atributos: int) -> List[Tuple[str, str]]:
    lista_atributos = list(atributos.items())
    if numero_atributos and len(lista_atributos) >= numero_atributos:
        lista_atributos = random.sample(lista_atributos, numero_atributos)
    return lista_atributos


def elegir_mejor_atributo(data: np.ndarray, indice_atributos, tipo_atributos, indice_clases, numero_atributos):
    lista_atributos = seleccionar_numero_atributos(tipo_atributos, numero_atributos)

    auxiliar = info_atributos(data, indice_atributos, tipo_atributos, lista_atributos, indice_clases)
    total_atributos_seleccionados, grupos_total, indice_grupos_total = auxiliar

    indice_seleccionado = total_atributos_seleccionados.index(min(total_atributos_seleccionados))

    nombre_seleccionado, tipo_seleccionado = lista_atributos[indice_seleccionado]
    grupo_seleccionado = grupos_total[indice_seleccionado]
    indice_grupo_seleccionado = indice_grupos_total[indice_seleccionado]

    return nombre_seleccionado, tipo_seleccionado, grupo_seleccionado, indice_grupo_seleccionado


def _entrenar(data, indice_atributos, tipo_atributos, indice_clases, m = None):
    if verificar_clase_unica(data, indice_clases):
        return ArbolDecision(etiqueta_clase=data[:, indice_clases][0])

    elif not tipo_atributos:
        return ArbolDecision(etiqueta_clase=clase_mas_frecuente(data[:, indice_clases]))

    else:
        mejor_atributo = elegir_mejor_atributo(data, indice_atributos, tipo_atributos, indice_clases, m)
        nombre, tipo, agrupados, indice = mejor_atributo

        nuevos_atributos = {k: valor for k, valor in tipo_atributos.items() if k != nombre}

        def generar_opciones():
            opciones = {}
            for indice_subgrupo, grupo in zip(indice, agrupados):
                opciones[indice_subgrupo] = _entrenar(grupo, indice_atributos, nuevos_atributos, indice_clases, m)
            return opciones

        valor_de_corte = data[:, indice_atributos[nombre]].mean() if tipo == "numerico" else None

        return ArbolDecision(
            atributo=nombre,
            tipo=tipo,
            opciones=generar_opciones(),
            valor_de_corte=valor_de_corte
        )


def predecir(arbol, instance):
    if arbol.etiqueta_clase or arbol.etiqueta_clase == 0:
        return arbol.etiqueta_clase

    try:
        if arbol.tipo == "nominal":
            sub_arbol = arbol.opciones[instance[arbol.atributo]]
        else:
            sub_arbol = arbol.opciones[instance[arbol.atributo] > arbol.valor_de_corte]
    except KeyError:
        sub_arbol = next(iter(arbol.opciones.values()))

    return predecir(sub_arbol, instance)
