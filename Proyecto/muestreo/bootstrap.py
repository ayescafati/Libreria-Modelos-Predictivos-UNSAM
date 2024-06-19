from numpy.random import randint


def generar_bootstraps(datos, n, semilla=randint(10000)):
    bootstraps = []
    for i in range(n):
        muestra = datos.sample(frac=1, replace=True, random_state = semilla + i) # en Matematica, Estadistica entendemos por "semilla" a aquel numero (o vector) utilizado para inicializar un generador de números pseudoaleatorios
        bootstraps.append(muestra)

    return bootstraps
