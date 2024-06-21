from util.procesamiento import procesar_dataset

if __name__ == "__main__":
    datasets = [
        ("HR_spectral_type", "data/HR_spectral_type.csv", "data/HR_spectral_type.json"),
        ("credit", "data/credit.csv", "data/credit.json"),
        ("spambase", "data/spambase.csv", "data/spambase.json"),
        ("wine", "data/wine.csv", "data/wine.json"),
        ("measurements", "data/measurements.csv", "data/measurements.json"),
    ]

    semilla = 5
    cardinal_k_subconjuntos = 5
    numero_arboles = 10
    nivel_verbosidad = 2
    paralelizar = False

    for nombre_dataset, ruta_dataset, ruta_atributos in datasets:
        procesar_dataset(nombre_dataset, ruta_dataset, ruta_atributos, semilla, cardinal_k_subconjuntos, numero_arboles, nivel_verbosidad, paralelizar)
