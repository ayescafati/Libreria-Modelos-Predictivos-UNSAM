import pandas as pd
import numpy as np


class MatrizConfusion:
    def __init__(self, resultados) -> None:
        self._resultados = resultados
        self._total = len(resultados)
        self._correcto = len(resultados[resultados['prediccion'] == resultados['class']])
        

        clases = np.unique(np.concatenate([resultados['class'].unique(), resultados['prediccion'].unique()]))
        self._matriz_confusion = pd.crosstab(pd.Categorical(resultados['class'], categories=clases), 
                               pd.Categorical(resultados['prediccion'], categories=clases),
                               rownames=['actual'],
                               colnames=['prediccion'],
                               dropna=False)
        
    def __add__(self, val) -> object:
        resultados_nuevos = pd.concat([self._resultados, val._resultados])
        return MatrizConfusion(resultados_nuevos)
    

    def __str__(self) -> str:
        return self._matriz_confusion.__repr__()

    def print(self: object) -> object:
        print(self._matriz_confusion)

    def verdaderos_positivos(self: object) -> object:
        vpositivos = pd.Series(np.diag(self._matriz_confusion.values), index=self._matriz_confusion.index)
        return vpositivos.rename_axis('VerdaderosPositivos')

    def verdaderos_negativos(self):
        vnegativos = [self._matriz_confusion.drop(index=[c], columns=[c]).values.sum() for c in self._matriz_confusion.index]
        return pd.Series(vnegativos, index=self._matriz_confusion.index).rename_axis('VerdaderosNegativos')

    def falsos_positivos(self):
        fpositivos = [self._matriz_confusion.loc[:, c].sum() - self._matriz_confusion.loc[c, c] for c in self._matriz_confusion.columns]
        return pd.Series(fpositivos, index=self._matriz_confusion.index).rename_axis('FalsosPositivos')

    def falsos_negativos(self):
        fnegativos = [self._matriz_confusion.loc[c, :].sum() - self._matriz_confusion.loc[c, c] for c in self._matriz_confusion.index]
        return pd.Series(fnegativos, index=self._matriz_confusion.index).rename_axis('FalsosNegativos')

    def exactitud(self):
        return self._correcto / self._total  # ratio entre las predicciones correctas (suma de verdaderos positivos y verdaderos negativos) y el total de predicciones 

    def error(self: object) -> object:
        return 1 - self.exactitud()

    def precision(self: object) -> object:
        vpositivos = self.verdaderos_positivos()
        fpositivos = self.falsos_positivos()
        precision = vpositivos / (vpositivos + fpositivos)
        return precision.rename_axis('Precision')

    def recalls(self: object) -> object:
        vpositivos = self.verdaderos_positivos()
        fnegativos = self.falsos_negativos()
        recalls = vpositivos / (vpositivos + fnegativos)
        return recalls.rename_axis('Recall')


    def especificidad(self: object) -> object:  # capacidad de nuestro estimador para discriminar los casos positivos, de los negativos
        vnegativos = self.verdaderos_negativos()
        fpositivos = self.falsos_positivos()
        especificidad = vnegativos / (vnegativos + fpositivos)
        return especificidad.rename_axis('Especificidad')

    def f_score(self, b):  # Notemos que: si b =1 (f1 score), se le da igual importancia a la precision y al recall; si b > 1, se le da mas importancia a la precision; si b < 1, se le da mas importancia al recall
        precisiones = self.precision()
        recuperaciones = self.recalls()
        f_score = ((1 + b**2) * precisiones * recuperaciones / (b**2 * precisiones + recuperaciones))
        return f_score.rename_axis('F-score')

    def macro_recall(self: object) -> object:
        return self.recalls().mean()

    def macro_precision(self: object) -> object:
        return self.precision().mean()

    def macro_especificidad(self: object) -> object:
        return self.especificidad().mean()

    def macro_f_score(self, b) -> object:
        return self.f_score(b).mean()
    
    def micro_f_score(self, b) -> object:
        precisiones = self.precision().sum()
        recuperaciones = self.recalls().sum()
        f_score = ((1 + b**2) * precisiones * recuperaciones / (b**2 * precisiones + recuperaciones))
        return f_score

    def micro_recall(self: object) -> object:
        vpositivos = self.verdaderos_positivos().sum()
        fnegativos = self.falsos_negativos().sum()
        recall = vpositivos / (vpositivos + fnegativos)
        return recall

    def micro_precision(self: object) -> object:
        vpositivos = self.verdaderos_positivos().sum()
        fpositivos = self.falsos_positivos().sum()
        precision = vpositivos / (vpositivos + fpositivos)
        return precision


    def mostrar(self, nivel_verbosidad = False) -> object:
        print("-" * 50)
        print(f"Exactitud: {self.exactitud():.3f} [Total: {self._total}, Correcto: {self._correcto}]")
        print(f"Macro Recall: {self.macro_recall():.3f}")

        if nivel_verbosidad:
            for clase, valor in self.recalls().items():
                print(f"\tRecall por clase {clase}: {valor:.3f}")

        print(f"Macro Precision: {self.macro_precision():.3f}")

        if nivel_verbosidad:
            for clase, valor in self.precision().items():
                print(f"\tPrecision por clase {clase}: {valor:.3f}")

        print(f"Macro Especificidad: {self.macro_especificidad():.3f}")

        if nivel_verbosidad:
            for clase, valor in self.especificidad().items():
                print(f"\tEspecificidad por clase {clase}: {valor:.3f}")

        for b in [2, 1, 0.5]:
            print(f"Macro F-score (ß = {b}): {self.macro_f_score(b):.3f}")

            if nivel_verbosidad:
                for clase, valor in self.f_score(b).items():
                    print(f"\tF-score (ß = {b}) por clase {clase}: {valor:.3f}")




    '''
    def mostrar(self, nivel_verbosidad = False):
        print("-"*50)
        print(f"exactitud: {self.exactitud():.3f} [Total: {self._total}, correcto: {self._correcto}]")
        print(f"Macro Recall: {self.macro_recall():.3f}")
        if nivel_verbosidad:
            for clase,  valor  in self.recalls().items():
                print(f"  Recall por clase {clase}  : {valor:.3f}")
        print(f"Macro Precision: {self.macro_precision():.3f}")
        if nivel_verbosidad:
            for clase,  valor  in self.precision().items():
                print(f"  Precision por clase {clase}  : {valor:.3f}")
        print(f"Macro Especificidad: {self.macro_especificidad():.3f}")
        if nivel_verbosidad:
            for clase,  valor  in self.especificidad().items():
                print(f"  Especificidad por clase {clase}  : {valor:.3f}")
        for b in [2, 1, 0.5]:
            print(f"Macro F-score (ß = {b}): {self.macro_f_score(b):.3f}")
            if nivel_verbosidad:
                for clase,  valor  in self.f_score(b).items():
                    print(f"  F-score (ß = {b}) por clase {clase}  : {valor:.3f}")

    '''