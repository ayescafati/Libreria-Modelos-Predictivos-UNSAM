# Proyecto: Librería de Modelos Predictivos con Árboles de Decisión y Random Forest

## Enunciado del Problema

El objetivo de este proyecto es desarrollar una librería que nos permita crear modelos predictivos mediante una versión simplificada del ensamble RandomForest. Buscamos entrenar un modelo con un conjunto de datos etiquetados, utilizando técnicas de aprendizaje supervisado, para predecir la variable objetivo. Inicialmente, nos enfocaremos en problemas de clasificación, con la posibilidad de extender la funcionalidad a problemas de regresión en el futuro.

## Introducción

Este proyecto implementa un clasificador de árbol de decisión y un bosque aleatorio en Python. Utilizamos principalmente el algoritmo ID3 para el árbol de decisión, con extensiones características del C4.5 que permiten el manejo de atributos continuos. Además, hemos desarrollado una implementación básica del bosque aleatorio. Nuestro clasificador puede procesar tanto atributos continuos como discretos, ofreciendo predicciones aceptables para conjuntos de datos de prueba.

## Árbol de Decisión

### Definición

Los árboles de decisión son modelos de aprendizaje automático que utilizamos para tareas de clasificación y regresión. Cada nodo interno del árbol representa una pregunta sobre una característica, cada rama representa el resultado de esa pregunta, y cada nodo hoja representa una clasificación o valor de predicción.

##Algoritmo ID3

El algoritmo ID3 se implementa en el código a través de la clase `ArbolDecision`. En esta clase, el método estático entrenar toma un conjunto de datos `df` en forma de matriz, un diccionario de tipos de atributos `tipo_atributos`, y opcionalmente el número de elementos a considerar `numero_elementos`, y devuelve un objeto `ArbolDecision` entrenado. Dentro del método `entrenar`, se realiza la construcción del árbol de decisión utilizando el enfoque del algoritmo ID3, donde se elige el mejor atributo para dividir los datos en cada paso basándose en la ganancia de información. Se ha usado recursividad para construir el árbol de manera eficiente.

En particular, la recursividad se utiliza en la función `_entrenar` de la misma clase, `ArbolDecision`. Esta función se encarga de construir el árbol de decisión de manera recursiva, dividiendo el conjunto de datos en subconjuntos basados en el mejor atributo en cada paso.


## Random Forest 

### Definición

Random Forest es una técnica de ensamble que utiliza múltiples árboles de decisión entrenados sobre diferentes subconjuntos de datos y características. La predicción final se obtiene mediante la votación mayoritaria (para clasificación) o el promedio (para regresión) de las predicciones individuales de los árboles.

##Construcción del Random Forest

La construcción del Random Forest se realiza mediante la clase `RandomForest`. El método `entrenar`, de esta clase, agarra un conjunto de datos de entrenamiento `conjunto_entrenamiento`, una lista de atributos `atributos`, el número de árboles `numero_arboles`, y (opcionalmente) el número de atributos a considerar `numero_atributos`, un objeto Pool para procesamiento paralelo pool, y una semilla para la aleatoriedad semilla. Este método utiliza el bootstrapping para generar múltiples conjuntos de datos de entrenamiento y entrena un conjunto de árboles de decisión utilizando la clase ArbolDecision. Luego, retorna un objeto RandomForest con los árboles entrenados.

###Bootstrapping

La técnica de bootstrapping se implementa en la función `generar_bootstraps` que toma un DataFrame df y un número `numero_muestras` que representa el número de muestras bootstrap  a generar, junto con una semilla para la aleatoriedad semilla. Este método utiliza la función `sample` de pandas para generar los bootstraps a partir del dataset de entrada, con reemplazo y manteniendo el mismo tamaño del dataset original.

###Selección Aleatoria de Características

La selección aleatoria de características se realiza en la función `seleccionar_numero_atributos`, que toma un diccionario de tipos de atributos atributos y un número `numero_atributos` que representa la cantidad de atributos a seleccionar aleatoriamente. Dentro de esta función, se crea una lista de atributos y se selecciona una muestra aleatoria de esta lista, garantizando que se seleccione el número correcto de atributos según la entrada.

###Combinación de Predicciones

La combinación de predicciones se implementa en el método `predecir` de la clase `RandomForest`. Este método toma una instancia `observaciones` y utiliza cada árbol de decisión entrenado en el Random Forest para realizar una predicción individual. Luego, combina estas predicciones utilizando el voto mayoritario para obtener la predicción final del Random Forest.

## Matriz de Confusión

### Definición

Una matriz de confusión es una herramienta que visualiza y resume el rendimiento de un algoritmo clasificador al comparar las predicciones del modelo con las clases reales de un conjunto de datos. Esta matriz desglosa el número de instancias que el modelo ha clasificado correctamente (verdaderos positivos y verdaderos negativos) frente a las instancias que ha clasificado incorrectamente (falsos positivos y falsos negativos). La denominación "matriz de confusión" proviene de su capacidad para mostrar cuánto confunde el modelo las diferentes clases al realizar sus predicciones. 

### Matriz de Confusión y Evaluación del Modelo

Para evaluar el rendimiento de nuestros modelos de aprendizaje automático, implementamos la clase `MatrizConfusion`, que nos permite calcular métricas clave y visualizar la matriz de confusión. La matriz de confusión es una herramienta fundamental en la evaluación de modelos de clasificación, ya que muestra el desempeño del modelo al predecir las clases verdaderas y falsas.

### Componentes de la Clase `MatrizConfusion`

#### Inicialización

Al crear una instancia de la clase `MatrizConfusion`, se proporciona un conjunto de resultados de predicción. Estos resultados suelen incluir la clase predicha (`prediccion`) y la clase verdadera (`clase`) para cada instancia en el conjunto de datos. La clase también calcula automáticamente el total de instancias y la cantidad de predicciones correctas.

#### Matriz de Confusión

La matriz de confusión se genera internamente utilizando la función `pd.crosstab` de pandas. Esta matriz muestra las predicciones realizadas por el modelo frente a las clases verdaderas en forma de una tabla, donde las filas rpresentan las clases verdaderas (`actual`) y las columnas representan las predicciones del modelo (`prediccion`).

#### Métricas de Evaluación

La clase `MatrizConfusion` proporciona una variedad de métodos para calcular métricas de evaluación del modelo:

- **Exactitud (`exactitud`):** Calcula la proporción de predicciones correctas sobre el total de predicciones.
- **Error (`error`):** Complemento de la exactitud, representa la proporción de predicciones incorrectas.
- **Precisión (`precision`):** Indica la proporción de predicciones positivas correctas respecto a todas las predicciones positivas.
- **Recall (`recalls`):** Muestra la proporción de instancias positivas que el modelo identifica correctamente.
- **Especificidad (`especificidad`):** Representa la proporción de instancias negativas que el modelo identifica correctamente.

#### F-scores y Métricas Agregadas

Además de las métricas básicas, la clase `MatrizConfusion` calcula F-scores para diferentes valores de beta (`b`) que nos permite ajustar el equilibrio entre precisión y recall. Luego, proporciona métricas agregadas como el promedio de recall (`macro_recall`), precisión (`macro_precision`), especificidad (`macro_especificidad`), y F-score (`macro_f_score`) para evaluar el desempeño general del modelo.

### Visualización y Análisis

La clase `MatrizConfusion` incluye el método `mostrar`, para visualizar la matriz de confusión y proporcionarnos las métricas de evaluación junto con información detallada sobre el rendimiento nuestros modelo. Estas capacidades nos facilitan el análisis exhaustivo del desempeño del modelo en términos de aciertos y errores en la predicción de clases positivas y negativas, lo que es fundamental para la toma de decisiones informadas sobre la calidad del modelo y posibles mejoras a implementar.

## Conclusión

En este proyecto, hemos desarrollado una librería para la creación de modelos predictivos utilizando árboles de decisión y una versión simplificada de Random Forest. Desde la definición de los árboles de decisión y un algoritmo ID3 mejorado, hasta la implementación de Random Forest con técnicas de bagging y selección aleatoria de características, nuestro enfoque ha sido proporcionar una herramienta flexible y eficaz para problemas de clasificación inicialmente, con el potencial de expandirse a problemas de regresión en el futuro.

La modularidad del código, reflejada en la estructura de nuestros módulos y clases, junto con la evaluación del rendimiento del Random Forest mediante métricas y la técnica de validación cruzada, han dotado a nuestro modelo de la capacidad de generalizar de manera aceptable y de manejar la complejidad de los datos de manera efectiva.


## Referencias

1. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor - An Introduction to Statistical Learning with Python (2023).
2. Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106.
3. Michael A. Nielsen. “Neural Networks and Deep Learning”, Determination Press, 2015.
4. François Chollet. “Deep Learning with Python”, Manning, 2017.
5. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436–444 (2015).
6. Hernández Orallo, Ramírez Quintana, Ferri Ramírez. Introducción a la Minería de Datos. Editorial Pearson – Prentice Hall. 2004
7. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
8. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
9. https://www.hbs.edu/research-computing-services/resources/compute-cluster/running-jobs/scaling-work.aspx
10. https://cienciadedatos.net/documentos/33_arboles_de_prediccion_bagging_random_forest_boosting#Ejemplo_regresi%C3%B3n
11.  https://tomaszgolan.github.io/introduction_to_machine_learning/markdown/introduction_to_machine_learning_02_dt/introduction_to_machine_learning_02_dt/#information-gain
12. https://www.ibm.com/topics/confusion-matrix
14. https://www.linkedin.com/advice/3/how-do-you-choose-between-simple-random-stratified-sampling?lang=es&originalSubdomain=es
15.  https://www.geeksforgeeks.org/no-quality-assurance-noqa-in-python/
16.   https://medium.com/@ramit.singh.pahwa/micro-macro-precision-recall-and-f-score-44439de1a044
