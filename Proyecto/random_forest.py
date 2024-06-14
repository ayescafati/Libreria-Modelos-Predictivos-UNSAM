import math
from arbol_decision import ArbolDecision
from scipy import stats
import statistics
from tqdm import tqdm
import random


class RandomForest:
    arboles = []
    entrenado = False

    def entrenar(self, dataset, n, cantidad_arboles):
        

        for _ in range(cantidad_arboles):
            bootstrap = dataset.bootstrap()
            t = ArbolDecision()
            # Ajuste del número de atributos a utilizar
            m = int(math.sqrt(n)) # Usamos raíz cuadrada del nro. total de características
            t.entrenar(bootstrap, m)
            self.arboles.append(t)

        self.entrenado = True

    def __call__(self, sample):
        assert self.entrenado

        votos = [t(sample) for t in self.arboles]
        votos_de_las_clases = [v[0] for v in votos]

        moda = stats.mode(votos_de_las_clases)[0][0][0]

        correct_confidences = [v[1] for v in votos if v[0] == moda]

        return moda, statistics.mean(correct_confidences)
