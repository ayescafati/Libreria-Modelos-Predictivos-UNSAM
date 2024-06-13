# Proyecto: Librería de Modelos Predictivos con Árboles de Decisión y Random Forest

## Enunciado del Problema

El objetivo de este proyecto es desarrollar una librería que nos permita crear modelos predictivos mediante una versión simplificada del ensamble RandomForest. Buscamos entrenar un modelo con un conjunto de datos etiquetados, utilizando técnicas de aprendizaje supervisado, para predecir la variable objetivo. Inicialmente, nos enfocaremos en problemas de clasificación, con la posibilidad de extender la funcionalidad a problemas de regresión en el futuro.

## Introducción

Este proyecto implementa un clasificador basado en árboles de decisión y un Random Forest en Python. Utilizamos el algoritmo C4.5, una mejora del algoritmo ID3, para construir los árboles de decisión. Este clasificador puede manejar tanto atributos continuos como discretos y proporciona predicciones para un conjunto de datos de prueba.

## Árbol de Decisión

### Definición

Los árboles de decisión son modelos de aprendizaje automático que utilizamos para tareas de clasificación y regresión. Cada nodo interno del árbol representa una pregunta sobre una característica, cada rama representa el resultado de esa pregunta, y cada nodo hoja representa una clasificación o valor de predicción.

### Algoritmo ID3

El algoritmo ID3 es una metodología que seguimos para construir árboles de decisión mediante los siguientes pasos:

1. *Selección del Atributo*: Utilizamos la entropía y la ganancia de información para seleccionar el atributo que mejor clasifica los datos en cada nivel del árbol. Realizamos esto en el modulo `ganancia_informacion.py`.
2. *División del Conjunto de Datos*: Dividimos el conjunto de datos en subconjuntos basados en los valores del atributo seleccionado. Realizamos esto en el modulo `ganancia_informacion.py`.
3. *Construcción Recursiva del Árbol*: Repetimos los pasos anteriores hasta que se cumple un criterio de parada, como alcanzar nodos puros o una profundidad máxima del árbol. Realizamos esto en la clas `ArbolDecision`
4. *Poda del Árbol*: En nuestra implementación del Random Forest, hemos estructurado el proceso de entrenamiento aprovechando técnicas como el bagging y la selección aleatoria de características. Estas estrategias son evidentes en nuestro código, como se observa en la función `obtener_subconjs_validacion_cruzada` del módulo `manejo_datos_csv`, donde dividimos el conjunto de datos para entrenamiento y validación cruzada, así como en la clase RandomForest del archivo random_forest, donde utilizamos el método entrenar que claramente aplica bagging al construir cada árbol con muestras aleatorias del conjunto de entrenamiento. El uso de bagging en nuestro Random Forest contribuye fuertemente a mitigar el riesgo de sobreajuste. Cada árbol individual dentro del ensemble se entrena con una muestra aleatoria del conjunto de datos, lo cual simplifica su estructura y reduce la tendencia al sobreajuste en comparación con árboles de decisión más profundos y complejos. En lugar de emplear técnicas de poda después del entrenamiento, confiamos en la diversidad y robustez de los árboles generados mediante el bagging y la aleatorización de características. Esta estrategia fortalece la capacidad del modelo para generalizar bien a nuevos datos, como se refleja en las métricas de desempeño obtenidas durante la validación cruzada en nuestro script. Al prescindir de la poda, mantenemos la capacidad predictiva y la eficiencia del algoritmo, asegurando que nuestro Random Forest pueda manejar una variedad de situaciones y conjuntos de datos de manera efectiva.

### Definición

Random Forest es una técnica de ensamble que utiliza múltiples árboles de decisión entrenados sobre diferentes subconjuntos de datos y características. La predicción final se obtiene mediante la votación mayoritaria (para clasificación) o el promedio (para regresión) de las predicciones individuales de los árboles.

# Construcción del Random Forest

## Bootstrapping
1. **Bootstrapping**: Realizamos un muestreo con reemplazo del conjunto de datos de entrenamiento para cada árbol. Esto se implementa en el método `entrenar` de la clase `RandomForest`.

## Selección Aleatoria de Características
2. **Selección Aleatoria de Características**: En cada división del árbol, consideramos un subconjunto aleatorio de características. Esto se implementa en el método `entrenar` de `RandomForest`, donde se selecciona un número aleatorio de características para cada árbol.

## Combinación de Predicciones
3. **Combinación de Predicciones**: Combinamos las predicciones de todos los árboles del bosque para obtener la predicción final. Esto se implementa en el método `__call__` de la clase `RandomForest`, donde se obtiene la moda de las predicciones de los árboles y se calcula la confianza promedio de las predicciones correctas.

# Implementación

El código del proyecto se divide en varios módulos y clases, cada uno con funciones específicas:

- **main.py**: Script principal que carga los datos, entrena los modelos y realiza predicciones utilizando el Random Forest.
  
- **arbol_decision.py**: Define la estructura y el entrenamiento de un árbol de decisión individual utilizando el algoritmo C4.5. Contiene la clase `ArbolDecision`.

- **random_forest.py**: Implementa la clase `RandomForest` que construye y maneja múltiples árboles de decisión para formar un Random Forest. Este archivo es fundamental para el entrenamiento y la predicción con Random Forest

- **funciones.py**: Contiene funciones auxiliares como `registrar`, utilizadas para el registro de mensajes durante el entrenamiento y la evaluación del modelo.

- **ganancia_informacion.py**: Contiene funciones y métodos para calcular la ganancia de información en atributos numéricos y categóricos, utilizados en la construcción de árboles de decisión y Random Forest.


# Implementación

En nuestra implementación del Random Forest, utilizamos la clase `RandomForest` ubicada en `random_forest.py`. Esta clase maneja la creación de un ensemble de árboles de decisión mediante la técnica de bagging. Cada árbol dentro del Random Forest se entrena con un subconjunto aleatorio de datos de entrenamiento, lo cual fomenta la diversidad y robustez del modelo al reducir la correlación entre los árboles individuales. Además, la selección aleatoria de características en cada árbol contribuye a esta diversidad, limitando la dependencia de ciertas variables y mejorando la capacidad de generalización del modelo final.

Para evaluar el rendimiento del Random Forest, hemos implementado funciones clave como `calcular_exactitud` y `f1_score` en `funciones.py`. Estas funciones nos permiten medir la precisión y el puntaje F1 del modelo respectivamente, evaluando su capacidad para clasificar conjuntos de datos diversos. Validamos nuestro enfoque mediante técnicas como la validación cruzada, asegurando una evaluación exhaustiva del modelo y fortaleciendo nuestra confianza en su habilidad para manejar la complejidad de los datos y generalizar correctamente a nuevas muestras. Todo esto se logra sin la necesidad de implementar técnicas adicionales de poda post-entrenamiento, evitando así añadir complejidad innecesaria al algoritmo.

Esta estructura modular y robusta no solo facilita la experimentación con diferentes configuraciones de hiperparámetros, sino que también sienta las bases para futuras extensiones del proyecto, como la integración de técnicas avanzadas de selección de características o la adaptación a problemas de regresión.

# Conclusión

En este proyecto hemos desarrollado una librería para la creación de modelos predictivos utilizando árboles de decisión y un ensemble simplificado de Random Forest. Desde la definición de los árboles de decisión y el algoritmo ID3 hasta la implementación del Random Forest con técnicas de bagging y selección aleatoria de características, nuestro enfoque ha sido proporcionar una herramienta flexible y eficaz para problemas de clasificación inicialmente, con potencial para expandirse a problemas de regresión en el futuro.

Destacamos la importancia de la modularidad del código, reflejada en la estructura de nuestros módulos y clases (`arbol_decision.py`, `random_forest.py`, `funciones.py`, `ganancia_informacion.py`), que permite ajustar hiperparámetros con facilidad y explorar nuevas funcionalidades. La evaluación del rendimiento del Random Forest mediante métricas como exactitud y F1 score, junto con técnicas de validación cruzada, le han dando a nuestro modelo la capacidad de generalizar aceptablemente y manejar la complejidad de los datos de manera efectiva.


## Referencias

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
2. Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106.
3. Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers.
4. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
5. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
6. https://uc-r.github.io/regression_trees
7. https://cienciadedatos.net/documentos/33_arboles_de_prediccion_bagging_random_forest_boosting#Ejemplo_regresi%C3%B3n
8. https://tomaszgolan.github.io/introduction_to_machine_learning/markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#information-gain
