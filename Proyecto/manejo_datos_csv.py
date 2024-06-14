'''

Manejo de atributos en la clase CsvDataset:

en la clase CsvDataset, los atributos numéricos son esos que se pueden 
transformar en números flotantes sin que haya ningún drama. 
Ahora, los atributos que no pueden convertirse en números flotantes, 
esos son considerados como atributos categóricos.

Fijate, en el método linea_a_dict, que es el que convierte cada línea del archivo 
CSV en un diccionario, ahí es donde hacemos esa distinción. Si cuando intentamos 
convertir un valor a número flotante no hay ningún quilombo, 
lo consideramos un atributo numérico. Pero si surge algún problema, 
ahí lo tratamos como un atributo categórico. 

'''

from random import choice
import funciones


class CsvDataset:
    nro_divisiones_validacion_cruzada = 10

    def __init__(self, nombre_archivo=None):
        self.nombre_archivo = nombre_archivo
        if nombre_archivo == None:
            self.encabezado = []
            self.items = []

        else:
            with open(nombre_archivo) as csvfile:
                lineas = csvfile.readlines()

            encabezado = lineas[0].split(',')
            
            self.encabezado = [x.replace('"', '').replace('\n', '').lower() for x in encabezado]

            self.items = []
            for una_linea in lineas[1::]:
                un_elemento = self.linea_a_dict(una_linea)
                self.items.append(un_elemento)

            assert len(self.items) == len(lineas) - 1  

    def linea_a_dict(self, line):
        line = line.split(',')
        un_elemento = {}
        for x, y in zip(line, self.encabezado):
            x = x.replace("'", '').replace('\n', '')
            try:
                un_elemento[y] = float(x)
                funciones.registrar(f"Adding numeric item to {y}")
            except ValueError:
                un_elemento[y] = x
                funciones.registrar(f"Adding categorical item to {y}")

        assert len(un_elemento) > 1  # Avoid empty

        return un_elemento

    def filtrar_dataset_categorico(self, attr, c):
        nuevo_dataset = CsvDataset()
        nuevo_dataset.encabezado = self.encabezado.copy()
        nuevo_dataset.items = [x for x in self.items if x[attr] == c]

        return nuevo_dataset

    def filtrar_dataset_numerico(self, attr, div):
        sub_datasets = {}
        for item in self.items:
            key = item[attr]
            if key not in sub_datasets:
                sub_datasets[key] = CsvDataset()
                sub_datasets[key].encabezado = self.encabezado.copy()
            sub_datasets[key].items.append(item)
        return sub_datasets

    def filtrar_atributos(self, attrs):
        nuevo_dataset = CsvDataset()
        nuevo_dataset.encabezado = attrs.copy()
        nuevo_dataset.items = [{x: i[x] for x in attrs} for i in self.items]

        return nuevo_dataset

    def bootstrap(self):
        '''
        Este método implementa el muestreo bootstrap, devolviendo un nuevo conjunto de datos
        donde cada elemento se elige aleatoriamente con reemplazo del conjunto de datos original.
        '''
        nuevo_dataset = CsvDataset()
        nuevo_dataset.encabezado = self.encabezado
        nuevo_dataset.items = [choice(self.items) for _ in range(len(self.items))]

        return nuevo_dataset

    def obtener_subconjs_validacion_cruzada(self, n):
        '''
        Este método se utiliza para dividir el conjunto de datos en subconjuntos y llevar a cabo 
        la validación cruzada. Retorna dos conjuntos de datos: uno para entrenamiento y otro para prueba, 
        excluyendo el n-ésimo pliegue como conjunto de prueba.

        El objeto `clases_por_elemento` es un diccionario que organiza los elementos según su etiqueta de clase. 
        Cada clave del diccionario representa una etiqueta de clase y su valor es una lista de elementos 
        que pertenecen a esa clase.

        Por ejemplo, si el conjunto de datos tiene clases como 'perro', 'gato', 'coche', etc., entonces 
        `clases_por_elemento` contendrá algo asi:

        ```python
        {
            'perro': [elemento1, elemento2, ...],
            'gato': [elemento3, elemento4, ...],
            'coche': [elemento5, elemento6, ...],
            ...
        }

        Luego, en el código, estos conjuntos de elementos se dividen en subconjuntos para llevar a cabo 
        la validación cruzada. Uno de los subconjs se asigna como conjunto de prueba, mientras que
        los demás se utilizan como conjunto de entrenamiento. El n-ésimo subconj se excluye 
        como conjunto de prueba.

        '''
        
        clases_por_elemento = {}
        for i in self.items:
            if i['class'] not in clases_por_elemento.keys():
                clases_por_elemento[i['class']] = [i]
            else:
                clases_por_elemento[i['class']].append(i)

        clases_por_elemento = {x: funciones.dividir_en_partes(clases_por_elemento[x], self.nro_divisiones_validacion_cruzada) for x in clases_por_elemento.keys()}

        entrenar_data = []
        test_data = []

        for k in clases_por_elemento.keys():
            for i in range(self.nro_divisiones_validacion_cruzada):
                if i == n:
                    entrenar_data += clases_por_elemento[k][i]
                else:
                    test_data += clases_por_elemento[k][i]

        entrenar_dataset = CsvDataset()
        entrenar_dataset.encabezado = self.encabezado
        entrenar_dataset.items = entrenar_data

        test_dataset = CsvDataset()
        test_dataset.encabezado = self.encabezado
        test_dataset.items = test_data

        return entrenar_dataset, test_dataset

    def remueve_atributo(self, attr):
        '''
        Este método devuelve un nuevo conjunto de datos que excluye el atributo especificado.
        '''
        nuevo_dataset = CsvDataset()
        nuevo_dataset.encabezado = self.encabezado.copy()
        nuevo_dataset.encabezado.remove(attr)
        nuevo_dataset.items = [{x: i[x] for x in nuevo_dataset.encabezado} for i in self.items]

        return nuevo_dataset

    def __iter__(self):
        return self.items.__iter__()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]
