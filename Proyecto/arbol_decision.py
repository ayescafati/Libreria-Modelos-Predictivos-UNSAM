import funciones
from ganancia_informacion import calcular_ganancia_info_num, calcular_ganancia_info_cat
from scipy import stats
from random import sample
import math

class ArbolDecision:
    es_hoja = False
    es_entrenado = False
    division = None
    cantidad_atributos = -1

    def entrenar(self, dataset, m):
        todos_atributos = dataset.encabezado.copy()
        todos_atributos.remove("class")
        # Utilizamos la raíz cuadrada del número de características
        self.cantidad_atributos = m
        self.atributos_a_usar = sample(todos_atributos, min(self.cantidad_atributos, len(todos_atributos)))
        self.atributos_a_usar.append("class")
        ganancia = self.encontrar_atributo_con_mayor_ganancia_info(dataset)

        if ganancia == 0:
            self.es_hoja = True

        elif self.es_numerico():
            sub_datasets = dataset.filtrar_dataset_numerico(self.atributo_divisor, self.division)
            self.siguiente = {k: ArbolDecision() for k in sub_datasets}
            for k, sub_dataset in sub_datasets.items():
                sub_dataset = sub_dataset.remueve_atributo(self.atributo_divisor)
                self.siguiente[k].cantidad_atributos = self.cantidad_atributos
                self.siguiente[k].entrenar(sub_dataset, m)

        else:  # Categorica y no hoja
            valores_validos = funciones.obtener_valores_posibles(dataset, self.atributo_divisor)
            self.siguiente = {}
            for valor in valores_validos:
                self.siguiente[valor] = ArbolDecision()
                self.siguiente[valor].cantidad_atributos = self.cantidad_atributos
                nuevo_dataset = dataset.filtrar_dataset_categorico(self.atributo_divisor, valor)
                nuevo_dataset = nuevo_dataset.remueve_atributo(self.atributo_divisor)
                self.siguiente[valor].entrenar(nuevo_dataset, m)

        clases = [muestra['class'] for muestra in dataset]
        self.clase_predicha = stats.mode(clases)[0]
        self.probabilidad = clases.count(self.clase_predicha) / len(dataset)

        self.es_entrenado = True

    def encontrar_atributo_con_mayor_ganancia_info(self, dataset):
        # Returns the ganancia

        mejor_ganancia_atributo = 0
        for atributo in self.atributos_a_usar:
            if atributo == 'class':
                continue
            if isinstance(dataset[0][atributo], float):
                ganancia, division = calcular_ganancia_info_num(dataset, atributo)
            else:
                ganancia = calcular_ganancia_info_cat(dataset, atributo)
                division = None

            if ganancia > mejor_ganancia_atributo:
                self.atributo_divisor = atributo
                mejor_ganancia_atributo = ganancia
                self.division = division

            funciones.registrar(f"Information ganancia for {atributo} = {ganancia}")

        return mejor_ganancia_atributo

    def es_numerico(self):
        return self.division is not None

    def __call__(self, sample, *args, **kwargs):  # Sobrecargamos al operador para que el objeto del arbol de decision se comporte como una funcion
        # *args nos da la posibilidad de pasar argumentos posicionales adicionales (opcional)
        # **kwargs nos da la posibilidad de pasar argumentos de palabras clave adicionales (opcional)
        
        if not self.es_entrenado:
            raise Exception("Árbol de decisiones aún sin entrenar. Por favor llame a dt.entrenar() antes de dt()")
        if self.es_hoja:
            return self.clase_predicha, self.probabilidad

        assert self.atributo_divisor is not None

        if self.es_numerico():
            key = sample[self.atributo_divisor]
            if key not in self.siguiente:
                return self.clase_predicha, self.probabilidad
            return self.siguiente[key](sample)
        else:
            clase_del_atributo = sample[self.atributo_divisor]
            if not clase_del_atributo in self.siguiente.keys():
                return self.clase_predicha, self.probabilidad
            else:
                return self.siguiente[clase_del_atributo](sample)

