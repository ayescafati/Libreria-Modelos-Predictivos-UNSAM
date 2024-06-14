
import csv
import math
import statistics
import os
from manejo_datos_csv import CsvDataset
from random_forest import RandomForest
import funciones

# Obtiene el directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define las rutas a los archivos de datos relativos al directorio actual
datasets = [CsvDataset(os.path.join(script_dir, "HR_spectral_type.csv")),
            CsvDataset(os.path.join(script_dir, "credit.csv")),
            CsvDataset(os.path.join(script_dir, "vertebra-column.csv")),
            CsvDataset(os.path.join(script_dir, "wine.csv"))]

f = open("registros.csv", 'w', newline='')
f.write("cantidad_arboles,dataset,exactitud,exactitud_sd,f1,f1_sd\n")

cantidad_arboles = 200

texto = '''
 ######     ##     ##   ##  #####     #####   ##   ##           #######   #####   ######   #######   #####   ######
  ##  ##   ####    ###  ##   ## ##   ##   ##  ### ###            ##   #  ##   ##   ##  ##   ##   #  ##   ##  # ## #
  ##  ##  ##  ##   #### ##   ##  ##  ##   ##  #######            ## #    ##   ##   ##  ##   ## #    #          ##
  #####   ##  ##   ## ####   ##  ##  ##   ##  #######            ####    ##   ##   #####    ####     #####     ##
  ## ##   ######   ##  ###   ##  ##  ##   ##  ## # ##            ## #    ##   ##   ## ##    ## #         ##    ##
  ##  ##  ##  ##   ##   ##   ## ##   ##   ##  ##   ##            ##      ##   ##   ##  ##   ##   #  ##   ##    ##
 #### ##  ##  ##   ##   ##  #####     #####   ##   ##           ####      #####   #### ##  #######   #####    ####
'''

print(texto)


for conjunto_prueba in datasets:
    # Contamos el número de filas del archivo
    with open(conjunto_prueba.nombre_archivo, 'r') as file:
        num_filas = sum(1 for line in file) - 1  # Resta 1 para excluir la línea del encabezado
    
    # Vemos si la cantidad de filas es menor o igual a 18
    if num_filas <= 18:
        raise ValueError("El archivo es demasiado corto. Debe tener al menos 18 lineas")

    print(f'Conjunto de prueba = {conjunto_prueba.nombre_archivo}\nCantidad de árboles = {cantidad_arboles}')
    print(f'Cantidad de filas en el archivo: {num_filas}')
    exactitudes = []
    f1_scores = []
    for subconjunto_validacion_cruzada in range(10):
        entrenar_dataset, _ = conjunto_prueba.obtener_subconjs_validacion_cruzada(subconjunto_validacion_cruzada)
        rf = RandomForest()

        rf.entrenar(entrenar_dataset, math.ceil(math.sqrt(len(entrenar_dataset.encabezado))), cantidad_arboles)

        predicciones = [rf(sample)[0] for sample in conjunto_prueba]
        esperados = [sample['class'] for sample in conjunto_prueba]

        exactitud = funciones.calcular_exactitud(predicciones, esperados)
        f1 = funciones.f1_score(predicciones, esperados)
        exactitudes.append(exactitud)
        f1_scores.append(f1)

        print(f"muestra {subconjunto_validacion_cruzada + 1}")
        print(f"exactitud: {exactitud:.5f}")
        print(f"F1 Score: {f1:.5f}")

    promedio_exactitud = statistics.mean(exactitudes)
    promedio_f1 = statistics.mean(f1_scores)
    desviacion_exactitud = statistics.stdev(exactitudes)
    desviacion_f1 = statistics.stdev(f1_scores)

    print(f"Promedio de exactitud entre los subconjuntos: {promedio_exactitud:.5f}")
    print(f"Promedio del puntaje F1 entre los subconjuntos: {promedio_f1:.5f}")
    print(f"Desviación estándar de la exactitud entre los subconjuntos: {desviacion_exactitud:.5f}")
    print(f"Desviación estándar del puntaje F1 entre los subconjuntos: {desviacion_f1:.5f}")
    
    f.write(f"{cantidad_arboles},{conjunto_prueba.nombre_archivo},{promedio_exactitud:.5f},{desviacion_exactitud:.5f},{promedio_f1:.5f},{desviacion_f1:.5f}\n")

    print("\n", "#"*80, "\n")

    # Guardar predicciones para el conjunto de datos de prueba
    ruta_archivo_predicciones = f'{os.path.splitext(conjunto_prueba.nombre_archivo)[0]}_predictions.csv'
    with open(ruta_archivo_predicciones, mode='w', newline='') as archivo_predicciones:
        escribir_predicciones = csv.writer(archivo_predicciones)
        escribir_predicciones.writerow(['Prediction', 'esperados'])

        for muestra in conjunto_prueba:
            clase_predicha, confianza = rf(muestra)
            escribir_predicciones.writerow([clase_predicha, muestra['class']])

f.close()


'''
# Obtiene el directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define las rutas a los archivos de datos relativos al directorio actual
datasets = [CsvDataset(os.path.join(script_dir, "HR_spectral_type.csv"))]
      ##      CsvDataset(os.path.join(script_dir, "Stars.csv")),
       ##     CsvDataset(os.path.join(script_dir, "vertebra-column.csv")),
       ##     CsvDataset(os.path.join(script_dir, "spambase.csv"))]

## CsvDataset(os.path.join(script_dir, "star_classification.csv")),

f = open("log.csv", 'w')
f.write("cantidad_arboles,dataset,exactitud,exactitud_sd,f1,f1_sd\n")

cantidad_arboles = 20

for conjunto_prueba in datasets:
    print(f'Conjunto de prueba = {conjunto_prueba.nombre_archivo}\nCantidad de árboles = {cantidad_arboles}')
    exactitudes = []
    f1_scores = []
    for subconjunto_validacion_cruzada in range(10):
        entrenar_dataset, _ = conjunto_prueba.obtener_subconjs_validacion_cruzada(subconjunto_validacion_cruzada)
        rf = RandomForest()

        rf.entrenar(entrenar_dataset, math.ceil(math.sqrt(len(entrenar_dataset.encabezado))), cantidad_arboles) #USAMOS RAIZ CUADRADA DEL NUMERO DE ATRIBUTOS 

        predicciones = [rf(sample)[0] for sample in conjunto_prueba]
        esperados = [sample['class'] for sample in conjunto_prueba]

        exactitud = funciones.calcular_exactitud(predicciones, esperados)
        f1 = funciones.f1_score(predicciones, esperados)
        exactitudes.append(exactitud)
        f1_scores.append(f1)

        print("exactitud:", exactitud)
        print("F1 Score:", f1)

    print("Promedio de exactitud entre los subconjuntos:", np.mean(exactitudes))
    print("Promedio del puntaje F1 entre los subconjuntos:", np.mean(f1_scores))
    print("Desviación estándar de la exactitud entre los subconjuntos:", np.std(exactitudes))
    print("Desviación estándar del puntaje F1 entre los subconjuntos:", np.std(f1_scores))
    f.write(f"{cantidad_arboles},{conjunto_prueba.nombre_archivo},{np.mean(exactitudes)},{np.std(exactitudes)},{np.mean(f1_scores)},{np.std(f1_scores)}\n")

    print("\n", "#"*80, "\n")

    # Guardar predicciones para el conjunto de datos de prueba
    ruta_archivo_predicciones = f'{os.path.splitext(conjunto_prueba.nombre_archivo)[0]}_predictions.csv'
    with open(ruta_archivo_predicciones, mode='w', newline='') as archivo_predicciones:
        escribir_predicciones = csv.writer(archivo_predicciones)
        escribir_predicciones.writerow(['Prediction', 'esperados'])

        for muestra in conjunto_prueba:
            clase_predicha, confianza = rf(muestra)
            escribir_predicciones.writerow([clase_predicha, muestra['class']])

f.close()






# Obtiene el directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define las rutas a los archivos de datos relativos al directorio actual
datasets = [CsvDataset(os.path.join(script_dir, "credit.csv")),
            CsvDataset(os.path.join(script_dir, "vertebra-column.csv")),
            CsvDataset(os.path.join(script_dir, "wine.csv"))]

f = open("log.csv", 'w')
f.write("cantidad_arboles,dataset,exactitud,exactitud_sd,f1,f1_sd\n")

cantidad_arboles = 1
while cantidad_arboles <= 260:
    for conjunto_prueba in datasets:
        print(f'Test dataset = {conjunto_prueba.nombre_archivo}\nNumber of trees = {cantidad_arboles}')
        exactitudes = []
        f1_scores = []
        for subconjunto_validacion_cruzada in range(10):
            train_dataset, _ = conjunto_prueba.get_folds(subconjunto_validacion_cruzada)
            rf = RandomForest()

            rf.train(train_dataset, math.ceil(math.sqrt(len(train_dataset.encabezado))), cantidad_arboles)

            predicciones = [rf(sample)[0] for sample in conjunto_prueba]
            esperados = [sample['class'] for sample in conjunto_prueba]

            exactitud = funciones.calcular_exactitud(predicciones, esperados)
            f1 = funciones.f1_score(predicciones, esperados)
            exactitudes.append(exactitud)
            f1_scores.append(f1)

            print("exactitud:", exactitud)
            print("F1 Score:", f1)

        print("Exactitud promedio entre folds:", statistics.mean(exactitudes))
        print("F1 Score promedio entre folds:", statistics.mean(f1_scores))
        print("Desviación estándar de la exactitud:", statistics.stdev(exactitudes))
        print("Desviación estándar de los F1 Scores:", statistics.stdev(f1_scores))
        f.write(f"{cantidad_arboles},{conjunto_prueba.nombre_archivo},{statistics.mean(exactitudes)},{statistics.stdev(exactitudes)},{statistics.mean(f1_scores)},{statistics.stdev(f1_scores)}\n")

        print("\n", "#"*80, "\n")

        # Guardar predicciones para el conjunto de datos de prueba
        ruta_archivo_predicciones = f'{os.path.splitext(conjunto_prueba.nombre_archivo)[0]}_predictions.csv'
        with open(ruta_archivo_predicciones, mode='w', newline='') as archivo_predicciones:
            escribir_predicciones = csv.writer(archivo_predicciones)
            escribir_predicciones.writerow(['Prediction', 'esperados'])

            for muestra in conjunto_prueba:
                clase_predicha, confianza = rf(muestra)
                escribir_predicciones.writerow([clase_predicha, sample['class']])

    cantidad_arboles += cantidad_arboles # vamos mas rapido

f.close()
'''

